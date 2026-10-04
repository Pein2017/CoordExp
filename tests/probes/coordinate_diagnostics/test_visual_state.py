"""Synthetic cached computation only; real frontend, hooks, publication and consumer."""
import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from probes.coordinate_diagnostics import visual_state as probe
from src.qwen.coordinate_policy import MedianPolicy
from src.qwen.native import NativeBatch
from src.qwen.untied_embeddings import SelectedDeltaOutputHead, SpecialTokenSelection


@pytest.fixture(autouse=True)
def bounded_threads():
    before = torch.get_num_threads(); torch.set_num_threads(2)
    yield
    torch.set_num_threads(before)


class FixtureEmbedding(torch.nn.Module):
    def forward(self, ids):
        return torch.ones(*ids.shape, 4, dtype=torch.bfloat16) * self.value


class FixtureBlock(torch.nn.Module):
    def forward(self, hidden_states, **kwargs):
        return hidden_states + 1


class FixtureLanguage(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([FixtureBlock() for _ in range(28)])
        self.norm = torch.nn.Identity()

    def forward(self, inputs_embeds, past_key_values, position_ids, deepstack_visual_embeds=None, visual_pos_masks=None):
        h = inputs_embeds
        for layer in self.layers:
            h = layer(h, past_key_values=past_key_values, position_ids=position_ids,
                attention_mask=None, position_embeddings=None)
        past_key_values.length += h.shape[1]
        return SimpleNamespace(last_hidden_state=self.norm(h), past_key_values=past_key_values)


class FixtureHead(SelectedDeltaOutputHead):
    def forward(self, h):
        pos = self.position
        scores = torch.full((1, h.shape[1], probe.COORD_START + 1000), -20., dtype=torch.bfloat16)
        token = self.cell['expected_ids'][pos]
        scores[:, :, token] = 8
        if pos == self.cell['position']:
            scores[:, :, probe.COORD_START + self.cell['target_box'][0]] += (h[:, :, 0] - 29) * .125
        return scores


class FixtureModel(torch.nn.Module):
    device = torch.device('cpu')
    config = SimpleNamespace()

    def __init__(self, packet):
        super().__init__()
        self.packet = packet
        self.embedding = FixtureEmbedding()
        self.embedding.shared_embed_delta = torch.zeros(1000, 4)
        self.language_model = FixtureLanguage()
        base = torch.nn.Linear(4, probe.COORD_START + 1000, bias=False, dtype=torch.bfloat16)
        with torch.no_grad():
            base.weight.fill_(1)
        base.requires_grad_(False)
        selection = SpecialTokenSelection(token_strings=tuple(f'<|coord_{i}|>' for i in range(1000)), token_ids=tuple(probe.COORD_IDS))
        self.head = FixtureHead(base, selection, torch.nn.Parameter(torch.zeros(1000, 4), requires_grad=False))

    def get_output_embeddings(self):
        return self.head

    def get_input_embeddings(self):
        return self.embedding

    def generate(self, **kwargs):
        ids = kwargs['input_ids'].clone(); raw, processed = [], []
        cache = SimpleNamespace(length=0); cache.get_seq_length = lambda: cache.length
        for pos in range(kwargs['max_new_tokens']):
            current = ids if pos == 0 else ids[:, -1:]
            self.embedding.value = {'clean': 1, 'target': 2, 'background': 3}[self.cell['input_region']]
            h = self.embedding(current)
            positions = torch.arange(cache.length, cache.length + current.shape[1]).reshape(1, -1)
            out = self.language_model(inputs_embeds=h, past_key_values=cache, position_ids=positions)
            self.head.position = pos; self.head.cell = self.cell
            scores = self.head(out.last_hidden_state[:, -1:])[:, -1].float()
            raw.append(scores)
            selected_scores = kwargs['logits_processor'](ids, scores); processed.append(selected_scores)
            selected = selected_scores.argmax(-1).reshape(1, 1)
            ids = torch.cat((ids, selected), 1)
            if int(selected) == kwargs['eos_token_id']:
                break
        return SimpleNamespace(sequences=ids, logits=tuple(raw), scores=tuple(processed))


class FixtureSession(probe.NativeSession):
    tokenizer = None
    processor = None
    break_clean = None
    break_final = None
    loads = []

    def __init__(self, checkpoint, packet):
        self.loads.append(checkpoint.name)
        model = FixtureModel(packet)
        self.q = SimpleNamespace(model=model, tokenizer=self.tokenizer)
        self.norm = MedianPolicy(model, probe.COORD_IDS)
        self.packet, self.delta = packet, None
        self.fixture_hidden_size = 4
        self.batches = {(image, region): NativeBatch(dict(input_ids=torch.tensor([r['prompt_token_ids']]),
            attention_mask=torch.ones(1, len(r['prompt_token_ids']), dtype=torch.long)), (r['request_id'],))
            for image, regions in packet['requests'].items() for region, r in regions.items()}
        self.composition = dict(compute='CPU_FIXTURE', research_model_loaded=False, research_forwards=0)

    def generate(self, cell, donor=None):
        self.q.model.cell = cell
        original = torch.autocast
        torch.autocast = lambda *args, **kwargs: original('cpu', enabled=False)
        try:
            r = super().generate(cell, donor)
        finally:
            torch.autocast = original
        if self.break_clean == cell['image_id'] and cell['phase'] == 'clean':
            r['median_steps'][0]['argmax'] += 1
        if self.break_final == (cell['image_id'], cell['region']) and cell['phase'] == 'final':
            r['_tensors']['normalized_h'] = r['_tensors']['normalized_h'].clone()
            r['_tensors']['normalized_h'][0] += 1
            r['capture']['tensor_identities']['normalized_h'] = probe.tensor_identity(r['_tensors']['normalized_h'])
        return r

    def close(self):
        self.q = self.norm = self.batches = None


@pytest.fixture(scope='module')
def prepared():
    from probes import rollout_row_credit as retained
    destination = Path(os.environ['VISUAL_STATE_CPU_OUTPUT'])
    q = retained.frontend(); assert q.model is None
    FixtureSession.tokenizer = q.tokenizer
    FixtureSession.processor = q.processor
    folder = destination / 'prepared'
    assert probe.main(['prepare', '--output', str(folder)]) == 0
    return folder, probe.a.load(folder / 'input-packet.json')


def test_actual_prepare_entry_publish_and_fresh_saved_consumer(prepared):
    folder, packet = prepared
    output = folder.parent / 'cpu-positive'
    FixtureSession.loads = []
    assert probe.main(['run', '--config', str(folder / 'native-proposal.json'), '--output', str(output)], cpu_factory=FixtureSession) == 0
    report = probe.a.load(output / 'readback.json')
    assert report['complete'] and report['completed_requests'] == 20 and report['generated_actions'] == 2550
    assert FixtureSession.loads == ['checkpoint-16']
    assert len(report['contrasts']) == 16
    assert all(row['control']['qualified'] for row in report['conditions'])
    assert all(r['construction_control'] == (r['phase'] == 'final') for r in report['contrasts'])
    assert all(r['physical_recovery_credit'] is False for c in report['conditions'] for r in c['analysis']['rows'])
    p = subprocess.run([sys.executable, '-m', 'probes.coordinate_diagnostics.visual_state', 'readback', '--output', str(output)],
        cwd=probe.ROOT, capture_output=True, text=True)
    (folder.parent / 'fresh-consumer.log').write_text(p.stdout + p.stderr)
    assert p.returncode == 0, p.stderr
    assert json.loads(p.stdout)['generated_actions'] == 2550
    with pytest.raises(FileExistsError):
        probe.main(['run', '--config', str(folder / 'native-proposal.json'), '--output', str(output)], cpu_factory=FixtureSession)
    assert not list(output.glob('*.partial-*'))


def positive_record(prepared, name):
    folder, packet = prepared
    path = folder.parent / 'cpu-positive' / (name + '.json')
    record = probe.a.load(path); record['artifact'] = probe.binding(path)
    return record, packet


@pytest.mark.parametrize('mutation', ['action', 'boundary', 'mask', 'prefix', 'budget', 'order'])
def test_frozen_matrix_and_mask_have_teeth(prepared, mutation):
    _, original = prepared
    p = copy.deepcopy(original)
    if mutation == 'action': p['conditions'][0]['position'] += 1
    if mutation == 'boundary': p['conditions'][8]['patch_layers'] = [13]
    if mutation == 'mask': p['sites']['351017']['target'][2] += 1
    if mutation == 'prefix': p['conditions'][0]['forced_actions']['13'] = 151649
    if mutation == 'budget': p['bounds']['actions'] += 1
    if mutation == 'order': p['conditions'][0], p['conditions'][1] = p['conditions'][1], p['conditions'][0]
    with pytest.raises(ValueError): probe.validate_conditions(p)
    assert sum(c['budget'] for c in original['conditions']) == 2550
    assert sum(len(c['forced_actions']) for c in original['conditions']) == 2450


def test_wrong_half_open_pixels_and_original_identity_fail(prepared, tmp_path):
    _, original = prepared
    p = copy.deepcopy(original); variant = p['pixels']['351017']['variants']['target']
    rgb = np.load(variant['array']['path'], allow_pickle=False)
    rgb[25, 258] = [63, 37, 35]  # just outside the declared half-open rectangle
    path = tmp_path / 'wrong.npy'; np.save(path, rgb, allow_pickle=False)
    variant['array'] = probe.binding(path)
    with pytest.raises(ValueError, match='half-open'): probe.verify_pixels(p)
    p = copy.deepcopy(original); p['original_media']['351017'] = '0' * 64
    with pytest.raises(ValueError, match='original media'): probe.verify_pixels(p)


def test_saved_donor_action_layer_identity_and_receiver_arguments(prepared):
    record, packet = positive_record(prepared, 'bottle-target-image')
    cell = next(c for c in packet['conditions'] if c['condition'] == 'bottle-target-early')
    tensors = probe.load_capture(record)
    probe.validate_donor(cell, (record, tensors), 1362)
    for change in ('image_id', 'position', 'prefix', 'layer', 'tensor'):
        r, t = copy.deepcopy(record), {k: v.clone() for k, v in tensors.items()}
        if change == 'image_id': r['image_id'] = 13348
        if change == 'position': r['capture']['position'] += 1
        if change == 'prefix': r['capture']['prefix_token_ids'][-1] += 1
        if change == 'layer': r['capture']['boundaries'] = [3, 13, 27]
        if change == 'tensor': t['residual_2'][0] += 1
        with pytest.raises(probe.CellInvalid): probe.validate_donor(cell, (r, t), 1362)
    model = FixtureModel(packet)
    cap = probe.ResidualCapture(model, cell, 1362, (record, tensors), hidden_size=4)
    cap.active, cap.calls, cap.evidence = True, cell['position'], {'consumed': []}
    hidden = torch.zeros(1, 1, 4, dtype=torch.bfloat16); before = hidden.clone()
    cache, positions, mask = object(), torch.tensor([[1375]]), torch.ones(1, 1)
    args, kwargs = cap.boundary(2, model.language_model.layers[3], (hidden,),
        dict(past_key_values=cache, position_ids=positions, attention_mask=mask))
    assert args[0] is not hidden and torch.equal(hidden, before)
    assert torch.equal(args[0][0, -1], tensors['residual_2'])
    assert kwargs['past_key_values'] is cache and kwargs['position_ids'] is positions and kwargs['attention_mask'] is mask
    cap.calls -= 1
    assert cap.boundary(2, model.language_model.layers[3], (hidden,), kwargs) is None
    multi = torch.arange(12, dtype=torch.bfloat16).reshape(1, 3, 4)
    old = multi.clone(); changed = probe.replace_current(multi, tensors['residual_2'])
    assert torch.equal(multi, old) and torch.equal(changed[:, :-1], old[:, :-1])
    assert torch.equal(changed[0, -1], tensors['residual_2'])


def test_shifted_action_actual_cache_and_exception_hook_removal(prepared):
    _, packet = prepared; cell = packet['conditions'][0]
    model = FixtureModel(packet); cap = probe.ResidualCapture(model, cell, 1362, hidden_size=4)
    modules = [model.embedding, model.language_model, model.head, model.language_model.norm,
               model.language_model.layers[3], model.language_model.layers[14]]
    before = [(len(m._forward_hooks), len(m._forward_pre_hooks)) for m in modules]
    with pytest.raises(RuntimeError, match='synthetic failure'):
        with cap:
            assert any(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in modules)
            raise RuntimeError('synthetic failure')
    assert before == [(len(m._forward_hooks), len(m._forward_pre_hooks)) for m in modules]
    cap.calls = cap.bound = cell['position']; cap.embedded = torch.tensor([[151648]])
    badcache = SimpleNamespace(get_seq_length=lambda: 1376)  # required1375
    with pytest.raises(probe.CellInvalid, match='cache'):
        cap.before(model.language_model, (), dict(past_key_values=badcache))
    with pytest.raises(probe.CellInvalid, match='shifted'):
        cap.bind(torch.ones(1, 1362 + 15, dtype=torch.long), torch.zeros(1, probe.COORD_START + 1000))


def test_full_vocabulary_endpoint_and_final_control_not_early_evidence(prepared):
    record, packet = positive_record(prepared, 'bottle-target-final')
    cell = next(c for c in packet['conditions'] if c['condition'] == record['condition'])
    summary = probe.scores(record, cell)
    for channel in ('raw', 'median'):
        values = record[channel + '_observations'][0]['coordinate_scores']
        assert summary[channel]['primary_target_minus_endpoint0'] == values[186] - values[0]
        assert summary[channel]['additional_fixed_contrasts']['s495_minus_s0'] == values[495] - values[0]
        expected = float(torch.exp(torch.logsumexp(torch.tensor(values).double(), 0) - summary[channel]['full_log_normalizer']))
        assert summary[channel]['coordinate_family_mass'] == pytest.approx(expected)
    rows = probe.a.load(Path(record['artifact']['path']).parent / 'readback.json')['conditions']
    for c in probe.contrasts(rows):
        assert c['construction_control'] == (c['phase'] == 'final')
        assert not c['effect_size_gate']
    capture = probe.load_capture(record); corrupted = {k: v.clone() for k, v in capture.items()}
    corrupted['raw'][3] += 1
    assert not probe.equality(capture, corrupted, ('raw', 'median', 'normalized_h'))['qualified']
    # Only x1 state/scores are the final donor transport control; free tails may differ.
    donor, _ = positive_record(prepared, 'bottle-target-image')
    records = {'bottle-target-image': donor}
    r = copy.deepcopy(record); r['token_ids'][-1] += 1
    assert probe.control(r, cell, records)['qualified']


def test_multi_free_likelihoods_semantic_tail_and_no_fabricated_iou(prepared):
    record, packet = positive_record(prepared, 'bottle-clean')
    cell = packet['conditions'][0]
    for pos in range(14, 19):
        assert record['raw_steps'][pos]['requested_token_id'] == record['token_ids'][pos]
        assert record['raw_steps'][pos]['pre_force_logprob'] == pytest.approx(record['raw_logprobs'][pos], abs=2e-5)
    bad = copy.deepcopy(record); bad['raw_steps'][14]['requested_token_id'] += 1
    with pytest.raises(ValueError, match='selected action'): probe.validate_record(bad, cell, packet)
    full = dict(cell, observations=list(range(14, 18)))
    analysis = probe.owner_entry.analyze(record, full, packet, FixtureSession.tokenizer)
    assert analysis['alignment'] == {'14': 'x1', '15': 'y1', '16': 'x2', '17': 'y2'}
    assert len(analysis['selected_row']['same_category_overlaps']) == sum(o['desc'] == 'bottle' for o in packet['images']['351017']['objects'])
    partial = copy.deepcopy(record); partial['token_ids'] = record['token_ids'][:15] + [packet['eos_id']]
    partial['text'] = FixtureSession.tokenizer.decode(partial['token_ids'], skip_special_tokens=False)
    partial['stop_reason'] = 'im_end'
    analysis = probe.owner_entry.analyze(partial, full, packet, FixtureSession.tokenizer)
    assert analysis['selected_row'] is None and analysis['selected_row_unavailable']
    assert 'designated_iou' not in analysis


def test_native_release_runtime_source_and_partial_output_fail_closed(prepared, tmp_path):
    folder, packet = prepared; proposal = folder / 'native-proposal.json'
    with pytest.raises(ValueError, match='exact clean lead release'): probe.load_packet(proposal)
    for name, mutate in [('runtime', lambda c: c['runtime'].update(torch='wrong')),
                         ('source', lambda c: c['producer_files'].update({probe.SOURCE_PATHS[0]: '0' * 64})),
                         ('packet', lambda c: c['input_packet'].update(sha256='0' * 64))]:
        c = copy.deepcopy(probe.a.load(proposal)); mutate(c); path = tmp_path / (name + '.json'); probe.a.write(path, c)
        with pytest.raises(ValueError): probe.load_packet(path, cpu=True)
    incomplete = tmp_path / 'partial'; incomplete.mkdir()
    probe.a.write(incomplete / 'conditions.json', [])
    with pytest.raises(ValueError, match='condition order'): probe.consume(incomplete, packet, FixtureSession.tokenizer)
    wrong = copy.deepcopy(packet); wrong['requests']['351017']['target']['crop'][2] -= 1
    with pytest.raises(ValueError, match='geometry'): probe.validate_conditions(wrong)
    wrong = copy.deepcopy(packet); wrong['cached_pipeline_files']['src/qwen/generation.py'] = '0' * 64
    path = tmp_path / 'stale-input.json'; probe.a.write(path, wrong)
    c = copy.deepcopy(probe.a.load(proposal)); c['input_packet'] = probe.binding(path)
    cfg = tmp_path / 'stale-proposal.json'; probe.a.write(cfg, c)
    with pytest.raises(ValueError, match='cached source'): probe.load_packet(cfg, cpu=True)


def test_resource_stop_precedes_any_fixture_load(prepared, monkeypatch):
    folder, _ = prepared
    monkeypatch.setattr(probe.readout, 'usage', lambda *args, **kwargs: dict(rss_bytes=probe.BOUNDS['rss_bytes'] + 1, retained_bytes=0))
    FixtureSession.loads = []
    out = folder.parent / 'cpu-resource-HOLD'
    assert probe.main(['run', '--config', str(folder / 'native-proposal.json'), '--output', str(out)], cpu_factory=FixtureSession) == 2
    assert not FixtureSession.loads
    assert probe.a.load(out / 'terminal.json')['counts']['attempted_requests'] == 0


def test_actual_native_initializer_processor_identity_seam(prepared, monkeypatch):
    from probes.rule_stability import runner
    _, packet = prepared
    def components(checkpoint):
        assert checkpoint.name == 'checkpoint-16'
        return SimpleNamespace(model=FixtureModel(packet), processor=FixtureSession.processor,
            tokenizer=FixtureSession.tokenizer), None, dict(compute='CPU_FIXTURE', research_model_loaded=False)
    monkeypatch.setattr(runner, 'native_components', components)
    session = probe.NativeSession(probe.PREVIOUS / 'checkpoint-16', packet)
    assert len(session.batches) == 6
    for (image, region), batch in session.batches.items():
        assert probe.batch_identity(batch) == packet['processor'][image][region]
    wrong = copy.deepcopy(packet)
    wrong['processor']['351017']['target']['tensors']['pixel_values']['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='executed processor'):
        probe.NativeSession(probe.PREVIOUS / 'checkpoint-16', wrong)
    # Synthetic CPU modules only; release references without invoking CUDA cleanup.
    session.q = session.norm = session.batches = None


def test_actual_shifted_action_partial_publication_and_donor_hold(prepared):
    folder, _ = prepared
    class Shifted(FixtureSession):
        def generate(self, cell, donor=None):
            if cell['condition'] == 'bottle-target-image':
                cell = dict(cell, position=cell['position'] + 1)
            return super().generate(cell, donor)
    out = folder.parent / 'cpu-partial-shift'
    assert probe.main(['run', '--config', str(folder / 'native-proposal.json'), '--output', str(out)], cpu_factory=Shifted) == 2
    report = probe.a.load(out / 'readback.json')
    states = {c['condition']: c['status'] for c in report['conditions']}
    assert states['bottle-target-image'] == 'attempted-invalid'
    assert states['bottle-target-final'] == states['bottle-target-early'] == states['bottle-target-middle'] == 'skipped-HOLD'
    assert states['person-target-middle'] == states['bottle-background-middle'] == 'completed'
    assert report['attempted_requests'] == 17 and report['completed_requests'] == 16
    partial = probe.a.load(out / 'bottle-target-image-invalid.json')
    assert 'pre-x1 token/cache' in partial['error']
    assert len(partial['partial']['selected_unconfirmed']) == 15
    assert not list(out.glob('*.partial-*'))


@pytest.mark.parametrize('failure', ['clean', 'final'])
def test_same_image_and_region_dependency_hold_on_actual_producer(prepared, failure):
    folder, packet = prepared; out = folder.parent / ('cpu-HOLD-' + failure)
    class Broken(FixtureSession):
        break_clean = 351017 if failure == 'clean' else None
        break_final = (351017, 'target') if failure == 'final' else None
    assert probe.main(['run', '--config', str(folder / 'native-proposal.json'), '--output', str(out)], cpu_factory=Broken) == 2
    report = probe.a.load(out / 'readback.json')
    states = {c['condition']: c['status'] for c in report['conditions']}
    assert all(states[c['condition']] == 'completed' for c in packet['conditions'] if c['image_id'] == 13348)
    if failure == 'clean':
        assert report['completed_requests'] == 11
        assert all(states[c['condition']] == 'skipped-HOLD' for c in packet['conditions'][2:] if c['image_id'] == 351017)
    else:
        assert report['completed_requests'] == 18
        assert states['bottle-target-early'] == states['bottle-target-middle'] == 'skipped-HOLD'
        assert states['bottle-background-early'] == states['bottle-background-middle'] == 'completed'
    assert not report['complete']
