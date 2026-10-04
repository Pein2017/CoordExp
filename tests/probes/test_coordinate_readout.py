"""Frozen cached-action, pre-force-channel and saved-consumer falsification."""
import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from probes import coordinate_readout as probe
from probes.rule_stability.__main__ import runtime_identity
from src.qwen.coordinate_policy import MedianPolicy
from src.qwen.native import NativeBatch


@pytest.fixture(autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def packet_fixture():
    conditions, requests = probe.definitions(), {}
    for condition in conditions:
        image, version = condition['image_id'], condition['history_version']
        saved = probe.a.load(probe.saved_path(version, image))
        analysis = probe.a.load(probe.saved_path(version, image, analysis=True))
        probe.verify_selected_rows(saved, analysis, version, image)
        expected = saved['token_ids'][:condition['budget']]
        prefix = expected[:-1].copy()
        for edit in condition['edits']:
            assert prefix[edit['position']] == edit['original']
            prefix[edit['position']] = edit['edited']
        condition.update(expected_ids=expected, forced_prefix=prefix,
            selectors=probe.selectors(analysis, condition['observations']),
            saved_raw_logprobs=saved['raw_logprobs'][:condition['budget']],
            saved_policy_logprobs=saved['policy_logprobs'][:condition['budget']])
        requests[str(image)] = dict(prompt_token_ids=saved['prompt_token_ids'])
    return dict(schema=probe.SCHEMA, conditions=conditions, requests=requests, images={},
        bounds=probe.BOUNDS, coordinate_ids=probe.COORD_IDS, eos_id=151645, object_start_id=151646,
        runtime=runtime_identity(), producer=probe.producer_identity(), model_loaded=False, bindings={}, payloads={})


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    owner = tmp_path / 'owner'
    owner.mkdir()
    monkeypatch.setattr(probe, 'OUTPUT', owner)
    directory = owner / 'prepared-fixture'
    directory.mkdir()
    packet = packet_fixture()
    probe.a.write(directory / 'static-fixture.json', dict(compute='CPU_FIXTURE', model_loaded=False))
    packet['static_geometry'] = probe.binding(directory / 'static-fixture.json')
    probe.a.write(directory / 'input-packet.json', packet)
    config = dict(schema=probe.SCHEMA, released=False, source_revision=probe.revision(),
        producer_files=packet['producer']['files'], input_packet=probe.binding(directory / 'input-packet.json'),
        runtime=packet['runtime'], bounds=probe.BOUNDS, retry='no_automatic_relaunch', output=str(owner / 'native-01'))
    probe.a.write(directory / 'native-proposal.json', config)
    return directory, packet


class SyntheticCachedModel:
    """Substitute model computation only; exercise the real generation consumer."""
    device = torch.device('cpu')
    config = SimpleNamespace()

    def __init__(self, packet):
        self.packet = packet
        head = SimpleNamespace(weight=torch.ones(probe.COORD_START + 1000, 1, dtype=torch.bfloat16),
            bias=None, selected_token_ids=torch.tensor(probe.COORD_IDS), shared_embed_delta=torch.zeros(1000, 1))
        self.norm = MedianPolicy(SimpleNamespace(get_output_embeddings=lambda: head,
            get_input_embeddings=lambda: SimpleNamespace(shared_embed_delta=torch.zeros(1000, 1))), probe.COORD_IDS)

    def generate(self, **kwargs):
        assert kwargs['generation_config'].do_sample is False and kwargs['repetition_penalty'] == 1
        assert kwargs['output_logits'] and kwargs['output_scores']
        ids, condition = kwargs['input_ids'].clone(), self.condition
        width = ids.shape[1]
        assert ids[0].tolist() == self.packet['requests'][str(condition['image_id'])]['prompt_token_ids']
        raw, processed = [], []
        for position in range(kwargs['max_new_tokens']):
            logits = torch.zeros(1, probe.COORD_START + 1000)
            token = condition['expected_ids'][position]
            if position == kwargs['max_new_tokens'] - 1 and condition['kind'].startswith(('zero-width-', 'normal-')):
                token = int(ids[0, width + condition['edits'][0]['position']])
            logits[0, token] = 8
            if position == kwargs['max_new_tokens'] - 1 and hasattr(self, 'competitor'):
                logits[0, self.competitor] = 7
            raw.append(logits)
            scores = kwargs['logits_processor'](ids, logits)
            processed.append(scores)
            selected = scores.argmax(-1).reshape(1, 1)
            ids = torch.cat((ids, selected), 1)
            if int(selected) == kwargs['eos_token_id']:
                break
        return SimpleNamespace(sequences=ids, logits=tuple(raw), scores=tuple(processed))


class FixtureSession(probe.NativeSession):
    loads = []

    def __init__(self, checkpoint, packet):
        self.loads.append(int(checkpoint.name.split('-')[1]))
        model = SyntheticCachedModel(packet)
        self.q = SimpleNamespace(model=model, tokenizer=SimpleNamespace(pad_token_id=151643,
            decode=lambda ids, **kwargs: ' '.join(map(str, ids))))
        self.delta, self.norm, self.packet = None, model.norm, packet
        self.batches = {image: NativeBatch(dict(input_ids=torch.tensor([r['prompt_token_ids']]),
                attention_mask=torch.ones(1, len(r['prompt_token_ids']), dtype=torch.long)), (f'fixture:{image}',))
            for image, r in packet['requests'].items()}
        self.composition = dict(compute='CPU_FIXTURE', model_loaded=False, model_forwards=0)

    def generate(self, condition):
        self.q.model.condition = condition
        original = torch.autocast
        torch.autocast = lambda *args, **kwargs: original('cpu', enabled=False)
        try:
            return super().generate(condition)
        finally:
            torch.autocast = original

    def close(self):
        self.q = self.norm = self.batches = None


def test_entry_to_fresh_consumer_and_exclusive_publication(prepared):
    directory, packet = prepared
    FixtureSession.loads = []
    output = probe.OUTPUT / 'cpu-positive'
    code = probe.main(['run', '--config', str(directory / 'native-proposal.json'), '--output', str(output)], cpu_factory=FixtureSession)
    terminal = probe.a.load(output / 'terminal.json')
    assert code == 0, probe.a.load(output / 'error.json') if (output / 'error.json').exists() else terminal
    assert FixtureSession.loads == [16, 0]
    assert terminal['counts'] == dict(checkpoint_loads=0, fixture_sessions=2, attempted_requests=16,
        completed_requests=16, generated_actions=3226, optimizer=0, backward=0, replay=0, training=0, warmup=0, exports=0)
    report = probe.a.load(output / 'readback.json')
    assert report['complete'] and report['compute'] == 'CPU_FIXTURE'
    assert all(r['natural_fidelity']['mismatch_count'] == 0 for r in report['conditions'] if r['kind'] == 'natural')
    fresh = subprocess.run([sys.executable, '-m', 'probes.coordinate_readout', 'readback', '--output', str(output)],
        cwd=probe.ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2'), text=True, capture_output=True)
    assert fresh.returncode == 0, fresh.stderr
    assert json.loads(fresh.stdout)['generated_actions'] == 3226
    with pytest.raises(FileExistsError):
        probe.main(['run', '--config', str(directory / 'native-proposal.json'), '--output', str(output)], cpu_factory=FixtureSession)
    condition = next(c for c in packet['conditions'] if c['kind'] == 'zero-width--1')
    record = probe.a.load(output / (condition['condition'].replace('/', '-') + '.json'))
    assert record['token_ids'][-1] == probe.COORD_START + 683
    assert record['policy_logprobs'][:-1] == [0.] * (condition['budget'] - 1)
    assert len(record['median_steps']) == 422 and report['contrasts']


def test_fidelity_hold_is_per_image_and_preserves_independent_cells(prepared):
    directory, _ = prepared
    class Corrupt(FixtureSession):
        def generate(self, condition):
            record = super().generate(condition)
            if condition['condition'] == 'B16/natural/7511':
                record['median_steps'][10]['argmax'] = 0
            return record
    output = probe.OUTPUT / 'cpu-hold'
    assert probe.main(['run', '--config', str(directory / 'native-proposal.json'), '--output', str(output)], cpu_factory=Corrupt) == 2
    report = probe.a.load(output / 'readback.json')
    held = [r['condition'] for r in report['conditions'] if r['status'] == 'HOLD']
    assert held == ['B16/zero-width--1/7511', 'B16/zero-width-+1/7511', 'B0/B16-history/7511']
    assert report['completed_requests'] == 13
    assert any(r['condition'] == 'B0/B16-history/13348' and r['status'] == 'completed' for r in report['conditions'])
    assert report['conditions'][1]['natural_fidelity']['mismatch_count'] == 1


def test_raw_and_median_pre_force_capture_and_action_offbyone():
    packet = packet_fixture()
    condition = copy.deepcopy(packet['conditions'][0])
    condition.update(budget=2, expected_ids=[probe.COORD_START, probe.COORD_START],
        forced_prefix=[probe.COORD_START], observations=[0, 1],
        selectors={'0': dict(role='x1', slot=0), '1': dict(role='x1', slot=0)})
    raw = probe.ReadoutProcessor(condition, packet, 3, force=False)
    median = probe.ReadoutProcessor(condition, packet, 3, force=True)
    logits = torch.zeros(1, probe.COORD_START + 1000)
    logits[0, probe.COORD_START + 1] = 8
    original, ids = logits.clone(), torch.tensor([[0, 0, 0]])
    assert raw(ids, logits) is logits
    transformed = logits.clone()
    transformed[0, probe.COORD_START + 1] = 2
    forced = median(ids, transformed)
    assert int(forced.argmax()) == probe.COORD_START
    assert torch.equal(logits, original) and forced.data_ptr() != transformed.data_ptr()
    assert raw.observations[0]['coordinate_scores'][1] == 8
    assert median.observations[0]['coordinate_scores'][1] == 2
    assert median.steps[0]['argmax'] == probe.COORD_START + 1
    assert median.steps[0]['pre_force_logprob'] < 0
    assert median(torch.tensor([[0, 0, 0, probe.COORD_START]]), transformed) is transformed
    with pytest.raises(ValueError, match='sequential'):
        median(torch.zeros(1, 6, dtype=torch.long), transformed)
    with pytest.raises(ValueError, match='sequential'):
        probe.ReadoutProcessor(condition, packet, 2, force=True)(ids, transformed)
    with pytest.raises(ValueError, match='nonfinite'):
        probe.capture_scores(torch.full_like(logits, float('nan')), packet)


def test_final_raw_winner_differs_from_actual_median_emission():
    packet = packet_fixture()
    condition = copy.deepcopy(packet['conditions'][0])
    condition.update(budget=2, expected_ids=[probe.COORD_START, probe.COORD_START + 1],
        forced_prefix=[probe.COORD_START], observations=[1], selectors={'1': dict(role='x1', slot=0)})
    session = FixtureSession(probe.PREVIOUS / 'checkpoint-16', packet)
    session.norm.head.weight[probe.COORD_START + 1] = 2
    session.q.model.competitor = probe.COORD_START + 2
    record = session.generate(condition)
    assert record['token_ids'][-1] == probe.COORD_START + 2
    assert record['raw_steps'][-1]['argmax'] == probe.COORD_START + 1
    assert record['median_steps'][-1]['argmax'] == probe.COORD_START + 2
    probe.validate_record(record, condition, packet)
    assert record['raw_steps'][-1]['requested_token_id'] == record['token_ids'][-1]
    assert record['raw_steps'][-1]['pre_force_logprob'] == record['raw_logprobs'][-1]


@pytest.mark.parametrize('mutation', ['prefix', 'edit_position', 'observations', 'budget', 'extra_request'])
def test_frozen_matrix_mutations_have_teeth(mutation):
    packet = packet_fixture()
    if mutation == 'prefix':
        packet['conditions'][3]['forced_prefix'][419] += 1
    elif mutation == 'edit_position':
        packet['conditions'][3]['edits'][0]['position'] += 1
    elif mutation == 'observations':
        packet['conditions'][3]['observations'] = [420]
    elif mutation == 'budget':
        packet['conditions'][0]['budget'] += 1
    else:
        packet['conditions'].append(copy.deepcopy(packet['conditions'][0]))
    with pytest.raises(ValueError):
        probe.validate_conditions(packet)


def test_release_source_runtime_and_resource_fail_closed(prepared, monkeypatch):
    directory, _ = prepared
    proposal = directory / 'native-proposal.json'
    with pytest.raises(ValueError, match='release'):
        probe.load_packet(proposal)
    original_config = probe.a.load(proposal)
    for name, mutation, message, cpu in [
        ('source', dict(released=True, source_revision='0' * 40), 'release', False),
        ('bounds', dict(bounds=dict(probe.BOUNDS, actions=3227)), 'resource', True),
        ('runtime', dict(runtime=dict(original_config['runtime'], python='wrong')), 'runtime', True)]:
        changed = directory / f'wrong-{name}.json'
        probe.a.write(changed, dict(original_config, **mutation))
        with pytest.raises(ValueError, match=message):
            probe.load_packet(changed, cpu=cpu)
    changed = directory / 'dirty-source.json'
    probe.a.write(changed, dict(original_config, released=True))
    original = subprocess.check_output
    monkeypatch.setattr(subprocess, 'check_output', lambda args, **kwargs:
        b' M probes/coordinate_readout.py\n' if args[:3] == ['git', 'status', '--porcelain=v1'] else original(args, **kwargs))
    with pytest.raises(ValueError, match='dirty'):
        probe.load_packet(changed)
    assert probe.resource_excess(dict(rss_bytes=probe.BOUNDS['rss_bytes'] + 1, retained_bytes=0), 0) == ['rss_bytes']
    assert probe.resource_excess(dict(rss_bytes=0, retained_bytes=0), 901) == ['wall_seconds']


def test_static_duplicate_and_bf16_addition_semantics():
    rows = torch.arange(3000, dtype=torch.float32).reshape(1000, 3) + 1
    rows[121] = rows[120]
    report = probe.geometry(rows, [(120, 121)])
    assert [120, 121] in report['exact_duplicate_groups']
    assert report['named_pairs'][0]['distance'] == 0
    base = torch.ones(1000, 2, dtype=torch.bfloat16)
    delta = torch.full((1000, 2), .003, dtype=torch.float32)
    assert not torch.equal((base + delta.to(torch.bfloat16)).float(), base.float() + delta)
    assert torch.equal(base + delta.to(torch.bfloat16), base)
