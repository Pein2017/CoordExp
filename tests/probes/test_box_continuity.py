"""CPU-only cached caller, literal action bookkeeping and saved-consumer checks."""
import copy
import json
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from probes import box_continuity as probe, coordinate_readout as readout
from probes.rule_stability.__main__ import runtime_identity
from src.qwen.coordinate_policy import MedianPolicy
from src.qwen.native import NativeBatch


@pytest.fixture(autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope='module')
def frontend():
    from probes import rollout_row_credit as retained
    q = retained.frontend()
    assert q.model is None
    return q


def packet_fixture():
    prior = probe.a.load(readout.OUTPUT / 'prepared-01/input-packet.json')
    cells = probe.definitions()
    for cell in cells:
        saved = probe.a.load(readout.saved_path(16, cell['image_id']))
        analysis = probe.a.load(readout.saved_path(16, cell['image_id'], analysis=True))
        cell.update(expected_ids=saved['token_ids'][:cell['budget']],
            saved_raw_logprobs=saved['raw_logprobs'][:cell['budget']],
            saved_policy_logprobs=saved['policy_logprobs'][:cell['budget']],
            selectors=readout.selectors(analysis, cell['observations']))
        cell['forced_actions'] = probe.forced_actions(cell)
    return dict(schema=probe.SCHEMA, conditions=cells, requests=prior['requests'], images=prior['images'],
        bounds=probe.BOUNDS, coordinate_ids=probe.COORD_IDS, eos_id=151645, object_start_id=151646,
        runtime=runtime_identity(), producer=probe.producer_identity(), model_loaded=False,
        bindings={}, payloads={}, cached_pipeline_files={}, trusted_owner=prior['trusted_owner'],
        reuse_conditions=[c for c in prior['conditions'] if c['kind'] in ('natural', 'copy--1', 'copy-+1')
                          and c['image_id'] == 351017 and c['version'] == 16])


class SyntheticCachedModel:
    """Substitute only scores; no neural model construction or forward."""
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
        ids, cell = kwargs['input_ids'].clone(), self.cell
        assert ids[0].tolist() == self.packet['requests'][str(cell['image_id'])]['prompt_token_ids']
        raw, processed = [], []
        for i in range(kwargs['max_new_tokens']):
            scores = torch.full((1, probe.COORD_START + 1000), -20.)
            token = cell['expected_ids'][i]
            if getattr(self, 'feedback', False) and cell['mode'] == 'free' and i == cell['edit_position'] + 2:
                token += cell['sign']
            scores[0, token] = 8
            if getattr(self, 'winner_split', False) and str(i) not in cell['forced_actions'] and token in probe.COORD_IDS:
                scores[0, token + 1] = 7
            if getattr(self, 'early_eos', False) and i == cell['edit_position'] + 1:
                scores[0, self.packet['eos_id']] = 10
            raw.append(scores)
            selected_scores = kwargs['logits_processor'](ids, scores)
            processed.append(selected_scores)
            selected = selected_scores.argmax(-1).reshape(1, 1)
            ids = torch.cat((ids, selected), 1)
            if int(selected) == kwargs['eos_token_id']:
                break
        return SimpleNamespace(sequences=ids, logits=tuple(raw), scores=tuple(processed))


class FixtureSession(probe.NativeSession):
    tokenizer = None
    loads = []

    def __init__(self, checkpoint, packet):
        self.loads.append(checkpoint.name)
        model = SyntheticCachedModel(packet)
        self.q = SimpleNamespace(model=model, tokenizer=self.tokenizer)
        self.delta, self.norm, self.packet = None, model.norm, packet
        self.batches = {image: NativeBatch(dict(input_ids=torch.tensor([r['prompt_token_ids']]),
            attention_mask=torch.ones(1, len(r['prompt_token_ids']), dtype=torch.long)), (f'fixture:{image}',))
            for image, r in packet['requests'].items()}
        self.composition = dict(compute='CPU_FIXTURE', model_loaded=False, model_forwards=0)

    def generate(self, cell):
        self.q.model.cell = cell
        original = torch.autocast
        torch.autocast = lambda *args, **kwargs: original('cpu', enabled=False)
        try:
            return super().generate(cell)
        finally:
            torch.autocast = original

    def close(self):
        self.q = self.norm = self.batches = None


@pytest.fixture
def prepared(tmp_path, monkeypatch, frontend):
    monkeypatch.setattr(probe, 'OUTPUT', tmp_path)
    FixtureSession.tokenizer = frontend.tokenizer
    packet = packet_fixture()
    session = FixtureSession(probe.PREVIOUS / 'checkpoint-16', packet)
    # Synthetic historical artifacts exercise the unchanged readout consumer.
    for old in packet['reuse_conditions']:
        cell = dict(image_id=351017, mode='fixture-clamped', expected_ids=old['expected_ids'],
            forced_actions={str(i): token for i, token in enumerate(old['forced_prefix'])},
            budget=old['budget'], observations=old['observations'], edit_position=16, sign=0)
        record = session.generate(cell)
        for channel in ('raw', 'median'):
            for obs in record[f'{channel}_observations']:
                obs['selector'] = old['selectors'][str(obs['position'])]
        name = old['condition'].replace('/', '-')
        path = tmp_path / f'{name}.json'
        probe.a.write(path, record)
        packet['bindings'][name] = probe.binding(path)
    session.close()
    directory = tmp_path / 'prepared'
    directory.mkdir()
    probe.a.write(directory / 'input-packet.json', packet)
    config = dict(schema=probe.SCHEMA, released=False, source_revision=probe.revision(),
        producer_files=packet['producer']['files'], input_packet=probe.binding(directory / 'input-packet.json'),
        runtime=packet['runtime'], bounds=probe.BOUNDS, retry='no_automatic_relaunch', output=str(tmp_path / 'native'))
    probe.a.write(directory / 'native-proposal.json', config)
    return directory, packet


def test_actual_entry_write_fresh_consumer_and_collision(prepared):
    directory, _ = prepared
    FixtureSession.loads = []
    output = probe.OUTPUT / 'cpu-positive'
    assert probe.main(['run', '--config', str(directory / 'native-proposal.json'), '--output', str(output)], cpu_factory=FixtureSession) == 0
    assert FixtureSession.loads == ['checkpoint-16']
    report = probe.a.load(output / 'readback.json')
    assert report['complete'] and report['completed_requests'] == 13 and report['generated_actions'] == 3390
    assert report['denominators'] == dict(within_box_x1_prescribed=4, preceding_row_x2_prescribed=2,
        within_box_x1_observed=4, preceding_row_x2_observed=2)
    assert len(report['matched_free_clamped']) == 6
    assert all(c['natural_fidelity']['mismatch_count'] == 0 for c in report['conditions'][:3])
    fresh = subprocess.run([sys.executable, '-m', 'probes.box_continuity', 'readback', '--output', str(output)],
        cwd=probe.ROOT, text=True, capture_output=True)
    assert fresh.returncode == 0, fresh.stderr
    assert json.loads(fresh.stdout)['generated_actions'] == 3390
    with pytest.raises(FileExistsError):
        probe.main(['run', '--config', str(directory / 'native-proposal.json'), '--output', str(output)], cpu_factory=FixtureSession)


def test_multiple_free_raw_policy_winners_and_all_action_likelihoods(prepared):
    _, packet = prepared
    cell = next(c for c in packet['conditions'] if c['image_id'] == 13348 and c['mode'] == 'free')
    session = FixtureSession(probe.PREVIOUS / 'checkpoint-16', packet)
    for i in (232, 233, 234):
        session.norm.head.weight[cell['expected_ids'][i]] = 2
    session.q.model.winner_split = True
    record = session.generate(cell)
    probe.validate_record(record, cell, packet)
    for i in (232, 233, 234):
        assert record['raw_steps'][i]['argmax'] == cell['expected_ids'][i]
        assert record['token_ids'][i] == cell['expected_ids'][i] + 1
        assert record['raw_steps'][i]['requested_token_id'] == record['token_ids'][i]
        assert record['raw_steps'][i]['selected_rank'] == 2
        assert record['raw_steps'][i]['pre_force_logprob'] == record['raw_logprobs'][i]
    wrong = copy.deepcopy(record)
    # Old last-free-only repair leaves earlier raw requests wrong.
    for i in (232, 233):
        wrong['raw_steps'][i]['requested_token_id'] = wrong['raw_steps'][i]['argmax']
    with pytest.raises(ValueError, match='selected action'):
        probe.validate_record(wrong, cell, packet)
    assert record['policy_logprobs'][231] == 0 and record['median_steps'][231]['pre_force_logprob'] < 0


def test_sparse_force_boundary_free_prefix_and_corruption(prepared):
    _, packet = prepared
    cell = next(c for c in packet['conditions'] if c['image_id'] == 13348 and c['mode'] == 'free')
    sparse = copy.deepcopy(cell)
    sparse['forced_actions']['234'] = cell['expected_ids'][234]
    session = FixtureSession(probe.PREVIOUS / 'checkpoint-16', packet)
    session.q.model.feedback = True
    record = session.generate(sparse)
    assert record['token_ids'][233] != sparse['expected_ids'][233]
    assert record['token_ids'][234] == sparse['expected_ids'][234]
    probe.validate_record(record, sparse, packet)
    wrong = copy.deepcopy(record)
    wrong['token_ids'][234] += 1
    with pytest.raises(ValueError, match='forced'):
        probe.validate_record(wrong, sparse, packet)


def test_natural_fidelity_rejects_forced_winner_and_holds_only_dependents(prepared, monkeypatch):
    directory, packet = prepared
    class Corrupt(FixtureSession):
        def generate(self, cell):
            record = super().generate(cell)
            if cell['mode'] == 'natural' and cell['image_id'] == 7511:
                record['median_steps'][10]['argmax'] = 0
            return record
    output = probe.OUTPUT / 'cpu-hold'
    assert probe.run(directory / 'native-proposal.json', output, cpu_factory=Corrupt) == 2
    report = probe.a.load(output / 'readback.json')
    assert report['completed_requests'] == 9
    assert len([c for c in report['conditions'] if c['status'] == 'HOLD']) == 4
    assert report['denominators']['within_box_x1_observed'] == 2
    assert report['denominators']['preceding_row_x2_observed'] == 2
    index = probe.a.load(output / 'conditions.json')
    held = next(e for e in index if e['status'] == 'HOLD')
    held.update(status='completed', filename='bogus.json')
    original = probe.a.load
    monkeypatch.setattr(probe.a, 'load', lambda path: index if Path(path) == output / 'conditions.json' else original(path))
    with pytest.raises(ValueError, match='dependency|dependent'):
        probe.consume(output, packet, FixtureSession.tokenizer)


def test_early_eos_and_actual_incomplete_structure_are_outcomes(prepared):
    _, packet = prepared
    cell = next(c for c in packet['conditions'] if c['image_id'] == 13348 and c['mode'] == 'free')
    session = FixtureSession(probe.PREVIOUS / 'checkpoint-16', packet)
    session.q.model.early_eos = True
    record = session.generate(cell)
    probe.validate_record(record, cell, packet)
    assert len(record['token_ids']) == 233 and record['token_ids'][-1] == packet['eos_id']
    analysis = probe.analyze(record, cell, packet, FixtureSession.tokenizer)
    assert not analysis['rows'] and analysis['malformed'] and analysis['alignment'].get('232') is None
    wrong = copy.deepcopy(record)
    wrong['stop_reason'] = 'length'
    with pytest.raises(ValueError, match='EOS'):
        probe.validate_record(wrong, cell, packet)


def test_parsed_boxes_owner_denominators_and_misaligned_role_suppression(prepared):
    _, packet = prepared
    session = FixtureSession(probe.PREVIOUS / 'checkpoint-16', packet)
    analyses = {}
    for cell in packet['conditions'][:3]:
        record = session.generate(cell)
        analyses[cell['image_id']] = probe.analyze(record, cell, packet, FixtureSession.tokenizer)
    normal = analyses[13348]['rows'][0]
    assert normal['box'] == [544, 632, 559, 682] and normal['owner_annotation_id'] == 191150
    assert normal['scale']['denominator_bins'] == 10
    zero = analyses[7511]['rows'][0]
    assert zero['width'] == 0 and not zero['valid'] and zero['scale']['denominator_bins'] is None
    bottle = analyses[351017]['rows']
    assert len(bottle) == 2 and all(r['owner_annotation_id'] is None and r['scale']['denominator_bins'] == 23 for r in bottle)
    assert set(normal['freely_generated_corner_displacements']) == {'y1', 'x2', 'y2'}
    bottle_cell = packet['conditions'][2]
    drift = session.generate(bottle_cell)
    drift['token_ids'][20] = 8987  # An actually changed free next-row description.
    drift['text'] = FixtureSession.tokenizer.decode(drift['token_ids'], skip_special_tokens=False)
    parsed_drift = probe.analyze(drift, bottle_cell, packet, FixtureSession.tokenizer)
    assert parsed_drift['rows'][1]['description'] != 'bottle'
    assert parsed_drift['rows'][1]['homologous'] is False
    assert parsed_drift['alignment'].get('24') is None
    corrupt = dict(drift, text=drift['text'] + 'corrupted')
    with pytest.raises(ValueError, match='token/text'):
        probe.analyze(corrupt, bottle_cell, packet, FixtureSession.tokenizer)
    left = session.generate(packet['conditions'][0])
    altered = copy.deepcopy(analyses[13348])
    altered['alignment']['232'][1] = 'x2'
    contrast = probe.aligned_contrast(left, left, analyses[13348], altered)
    assert 232 in contrast['unaligned_positions']
    assert all(o['position'] != 232 for o in contrast['score_comparisons']['median'])


def test_family_mass_separate_from_conditional_tv_w1_and_ties():
    o = dict(coordinate_scores=[0.] * 1000, full_log_normalizer=math.log(2000), argmax=10, top5_noncoordinate=[])
    changed = dict(o, full_log_normalizer=math.log(4000))
    delta = probe.score_difference(o, changed)
    assert delta['coordinate_conditional_TV'] == delta['coordinate_conditional_W1_bins'] == 0
    assert delta['family_mass_left'] == pytest.approx(.5) and delta['family_mass_right'] == pytest.approx(.25)
    step = dict(requested_token_id=probe.COORD_START, selected_rank=1, selected_tie_count=1000, pre_force_logprob=-8., forced=False)
    summary = probe.score_summary(o, step)
    assert summary['top2_margin'] == 0 and summary['coordinate_winner_ties'] == 1000
    point, shifted = dict(o, coordinate_scores=[-1000.] * 1000), dict(o, coordinate_scores=[-1000.] * 1000)
    point['coordinate_scores'][10] = shifted['coordinate_scores'][20] = 0
    assert probe.score_difference(point, shifted)['coordinate_conditional_W1_bins'] == 10


@pytest.mark.parametrize('mutation', ['budget', 'count', 'edit', 'force', 'clamp', 'context'])
def test_frozen_matrix_budget_fail_closed(mutation):
    packet = packet_fixture()
    if mutation == 'budget':
        packet['conditions'][0]['budget'] += 1
    elif mutation == 'count':
        packet['conditions'].append(copy.deepcopy(packet['conditions'][0]))
    elif mutation == 'edit':
        packet['conditions'][3]['edited'] += 1
    elif mutation == 'force':
        packet['conditions'][3]['forced_actions']['231'] += 1
    elif mutation == 'clamp':
        packet['conditions'][4]['clamp_positions'] = [232]
    else:
        packet['requests']['7511']['prompt_token_ids'] += [0] * 2000
    with pytest.raises(ValueError):
        probe.validate_conditions(packet)


def test_unreleased_source_runtime_and_resource_fail_closed(prepared, monkeypatch):
    directory, _ = prepared
    proposal = directory / 'native-proposal.json'
    with pytest.raises(ValueError, match='release'):
        probe.load_packet(proposal)
    config = probe.a.load(proposal)
    wrong = directory / 'wrong-runtime.json'
    probe.a.write(wrong, dict(config, runtime=dict(config['runtime'], python='wrong')))
    with pytest.raises(ValueError, match='runtime'):
        probe.load_packet(wrong, cpu=True)
    released = directory / 'dirty.json'
    probe.a.write(released, dict(config, released=True))
    original = subprocess.check_output
    monkeypatch.setattr(subprocess, 'check_output', lambda args, **kwargs:
        b' M AGENTS.md\n' if args[:3] == ['git', 'status', '--porcelain=v1'] else original(args, **kwargs))
    with pytest.raises(ValueError, match='dirty'):
        probe.load_packet(released)
    monkeypatch.setattr(readout, 'usage', lambda *args, **kwargs: dict(rss_bytes=probe.BOUNDS['rss_bytes'] + 1, retained_bytes=0))
    assert probe.resource_excess(directory, __import__('time').monotonic(), native=False)[1] == ['rss_bytes']


def test_reused_bottle_scores_must_match_fresh_natural(prepared):
    _, packet = prepared
    cell = next(c for c in packet['conditions'] if c['image_id'] == 351017 and c['mode'] == 'natural')
    session = FixtureSession(probe.PREVIOUS / 'checkpoint-16', packet)
    record = session.generate(cell)
    analysis = probe.analyze(record, cell, packet, FixtureSession.tokenizer)
    assert probe.reuse_bottle(packet, record, analysis, FixtureSession.tokenizer)['status'] == 'compatible'
    changed = copy.deepcopy(record)
    changed['median_observations'][2]['coordinate_scores'][23] += .001
    result = probe.reuse_bottle(packet, changed, analysis, FixtureSession.tokenizer)
    assert result['status'] == 'HOLD' and not result['contrasts']
    identity = packet['bindings']['B16-copy--1-351017']
    altered = copy.deepcopy(packet)
    altered['bindings']['B16-copy--1-351017'] = dict(identity, sha256='0' * 64)
    # Consumer rejects changed historical bytes even if parsed boxes stay identical.
    with pytest.raises(ValueError, match='reused artifact identity'):
        probe.consume(probe.OUTPUT, altered, FixtureSession.tokenizer)
