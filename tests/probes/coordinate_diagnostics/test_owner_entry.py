"""Model-free cached generation and tiny output-head instrumentation checks."""
import copy
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from probes.coordinate_diagnostics import owner_entry as probe
from src.qwen.native import NativeBatch
from src.qwen.coordinate_policy import MedianPolicy
from src.qwen.untied_embeddings import SelectedDeltaOutputHead, SpecialTokenSelection


@pytest.fixture(autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


class FixtureHead(SelectedDeltaOutputHead):
    """Substitute only native score computation; exercise real Module hook dispatch."""
    def forward(self, hidden):
        return self.current_scores.unsqueeze(1).expand(1, hidden.shape[1], -1).to(torch.bfloat16)


class FixtureModel:
    device = torch.device('cpu')
    config = SimpleNamespace()

    def __init__(self, packet):
        self.packet = packet
        base = torch.nn.Linear(2, probe.COORD_START + 1000, bias=False, dtype=torch.bfloat16)
        with torch.no_grad():
            base.weight.fill_(1)
        base.requires_grad_(False)
        selection = SpecialTokenSelection(token_strings=tuple(f'<|coord_{i}|>' for i in range(1000)), token_ids=tuple(probe.COORD_IDS))
        self.head = FixtureHead(base, selection, torch.nn.Parameter(torch.zeros(1000, 2), requires_grad=False))
        self.input_head = SimpleNamespace(shared_embed_delta=torch.zeros(1000, 2))
        self.norm = MedianPolicy(self, probe.COORD_IDS)
        self.head_calls = 0

    def get_output_embeddings(self):
        return self.head

    def get_input_embeddings(self):
        return self.input_head

    def generate(self, **kwargs):
        ids, cell = kwargs['input_ids'].clone(), self.cell
        raw, processed = [], []
        for i in range(kwargs['max_new_tokens']):
            scores = torch.full((1, probe.COORD_START + 1000), -20.)
            token = cell['expected_ids'][i]
            scores[0, token] = 8
            if getattr(self, 'winner_split', False) and str(i) not in cell['forced_actions'] and token in probe.COORD_IDS:
                scores[0, token + 1] = 7
            if getattr(self, 'early_eos', False) and i == cell['observations'][0] + 1:
                scores[0, self.packet['eos_id']] = 10
            self.head.current_scores = scores
            hidden = torch.ones(1, ids.shape[1] if i == 0 else 1, 2, dtype=torch.bfloat16)
            scores = self.head(hidden)[:, -1].float()
            self.head_calls += 1
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
    break_image = None

    def __init__(self, checkpoint, packet):
        self.loads.append(checkpoint.name)
        model = FixtureModel(packet)
        self.q = SimpleNamespace(model=model, tokenizer=self.tokenizer)
        self.delta, self.norm, self.packet = None, model.norm, packet
        self.batches = {image: NativeBatch(dict(input_ids=torch.tensor([r['prompt_token_ids']]),
            attention_mask=torch.ones(1, len(r['prompt_token_ids']), dtype=torch.long)), (f'fixture:{image}',))
            for image, r in packet['requests'].items()}
        self.composition = dict(compute='CPU_FIXTURE', model_loaded=False, model_forwards=0)
        self.companion_rows = probe.runtime_rows(model, self.norm)
        self.companion_status = dict(status='supported', primary_untouched=True, operation='CPU_FIXTURE_HEAD')

    def generate(self, cell):
        self.q.model.cell = cell
        original = torch.autocast
        torch.autocast = lambda *args, **kwargs: original('cpu', enabled=False)
        try:
            record = super().generate(cell)
        finally:
            torch.autocast = original
        if self.break_image == cell['image_id'] and cell['mode'] == 'natural':
            record['median_steps'][0]['argmax'] += 1
        return record

    def close(self):
        self.q = self.norm = self.batches = None


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, 'OUTPUT', tmp_path)
    from probes import rollout_row_credit as retained
    q = retained.frontend()
    assert q.model is None
    FixtureSession.tokenizer = q.tokenizer
    directory = tmp_path / 'prepared'
    assert probe.main(['prepare', '--output', str(directory)]) == 0
    return directory, probe.a.load(directory / 'input-packet.json')


def test_actual_prepare_entry_write_saved_consumer_and_collision(prepared):
    directory, _ = prepared
    FixtureSession.loads = []
    output = probe.OUTPUT / 'cpu-positive'
    assert probe.main(['run', '--config', str(directory / 'native-proposal.json'), '--output', str(output)], cpu_factory=FixtureSession) == 0
    assert FixtureSession.loads == ['checkpoint-16']
    report = probe.a.load(output / 'readback.json')
    assert report['complete'] and report['completed_requests'] == 10 and report['generated_actions'] == 841
    assert report['denominators'] == dict(prescribed=dict(x1_cued=4, uncued=3, teacher_path=3),
                                         observed=dict(x1_cued=4, uncued=3, teacher_path=3))
    assert all(c['natural_fidelity']['mismatch_count'] == 0 for c in report['conditions'][:2])
    assert report['companion']['status'] == 'complete' and len(report['companion']['projections']) == 40
    fresh = subprocess.run([sys.executable, '-m', 'probes.coordinate_diagnostics.owner_entry', 'readback', '--output', str(output)],
        cwd=probe.ROOT, text=True, capture_output=True)
    assert fresh.returncode == 0, fresh.stderr
    assert json.loads(fresh.stdout)['generated_actions'] == 841
    with pytest.raises(FileExistsError):
        probe.main(['run', '--config', str(directory / 'native-proposal.json'), '--output', str(output)], cpu_factory=FixtureSession)


@pytest.fixture(scope='module')
def packet():
    from probes import rollout_row_credit as retained
    q = retained.frontend()
    FixtureSession.tokenizer = q.tokenizer
    return probe.assemble_packet(q)


def test_multiple_free_actual_raw_selection_and_hook_cleanup(packet):
    cell = packet['conditions'][0]
    session = FixtureSession(probe.PREVIOUS / 'checkpoint-16', packet)
    with torch.no_grad():
        for i in cell['observations']:
            session.norm.head.weight[cell['expected_ids'][i]].fill_(2)
    session.companion_rows = probe.runtime_rows(session.q.model, session.norm)
    session.q.model.winner_split = True
    record = session.generate(cell)
    probe.validate_record(record, cell, packet)
    for i in cell['observations']:
        assert record['raw_steps'][i]['argmax'] == cell['expected_ids'][i]
        assert record['token_ids'][i] == cell['expected_ids'][i] + 1
        assert record['raw_steps'][i]['requested_token_id'] == record['token_ids'][i]
    assert not session.q.model.head._forward_hooks and session.q.model.head_calls == 19
    wrong = copy.deepcopy(record)
    wrong['raw_steps'][14]['requested_token_id'] = cell['expected_ids'][14]
    with pytest.raises(ValueError, match='action'):
        probe.validate_record(wrong, cell, packet)
    def fails(**kwargs):
        raise RuntimeError('fixture generation failure')
    session.q.model.generate = fails
    with pytest.raises(RuntimeError, match='fixture generation failure'):
        session.generate(cell)
    assert not session.q.model.head._forward_hooks
    assert session.failure_evidence['condition'] == cell['condition']
    assert session.failure_evidence['confirmed_token_ids'] is None


def test_target_margins_family_and_full_vocabulary_denominators():
    scores = torch.zeros(1, probe.COORD_START + 1000)
    target = probe.COORD_START + 10
    scores[0, target] = 3
    scores[0, target + 1] = 2
    scores[0, 8987] = 4
    m = probe.target_metrics(scores, target)
    assert m['coordinate_rank'] == 1 and m['full_rank'] == 2
    assert m['target_minus_best_other_coordinate'] == 1 and m['target_minus_best_other_full'] == -1
    o = probe.readout.capture_scores(scores, dict(eos_id=151645, object_start_id=151646))
    summary = probe.continuity.score_summary(o, dict(requested_token_id=target, selected_rank=2,
        selected_tie_count=1, pre_force_logprob=-5., forced=True))
    assert summary['coordinate_family_mass'] < .01
    assert torch.tensor(o['coordinate_scores']).softmax(-1).sum() == pytest.approx(1)
    scores[0, target + 1] = 3
    assert probe.target_metrics(scores, target)['coordinate_ties'] == 2


def test_true_shared_hidden_fp32_precision_sensitivity_and_factor_binding():
    base = torch.nn.Linear(2, probe.COORD_START + 1000, bias=False, dtype=torch.bfloat16)
    with torch.no_grad():
        base.weight[:, 0].fill_(-2)
        base.weight[:, 1].zero_()
        base.weight[probe.COORD_START + 10:probe.COORD_START + 12].fill_(1)
    base.requires_grad_(False)
    delta = torch.zeros(1000, 2)
    delta[10] = torch.tensor([.001, -.001])
    delta[11] = torch.tensor([.002, -.002])
    selection = SpecialTokenSelection(token_strings=tuple(f'<|coord_{i}|>' for i in range(1000)), token_ids=tuple(probe.COORD_IDS))
    head = SelectedDeltaOutputHead(base, selection, torch.nn.Parameter(delta, requires_grad=False))
    model = SimpleNamespace(get_output_embeddings=lambda: head,
        get_input_embeddings=lambda: SimpleNamespace(shared_embed_delta=torch.zeros_like(delta)))
    norm = MedianPolicy(model, probe.COORD_IDS)
    transform = norm.generation_transform()
    h = torch.tensor([[[1., 0.]]], dtype=torch.bfloat16)
    cell = dict(observations=[0])
    cap = probe.HeadCapture(cell, 1, transform, 'fixture')
    original = h.clone()
    baseline = head(h)
    handle = head.register_forward_hook(cap.hook)
    observed = head(h)
    handle.remove()
    assert torch.equal(h, original) and torch.equal(baseline, observed)
    cap.bind(torch.tensor([[42]]), observed[:, -1].float())
    normalized = transform(torch.tensor([[42]]), observed[:, -1].float())
    captured = cap.finish(transform, [probe.COORD_START + 10], [dict(coordinate_scores=normalized[0, probe.COORD_START:].tolist())])
    rows = probe.runtime_rows(model, norm)
    precision = torch.backends.fp32_precision
    mkldnn_precision = torch.backends.mkldnn.fp32_precision
    shadow = probe.shadow_scores(torch.tensor(captured['hidden'][0], dtype=torch.bfloat16), rows['base'], rows['delta'], cap.factors)
    assert torch.backends.fp32_precision == precision and torch.backends.mkldnn.fp32_precision == mkldnn_precision
    assert baseline[0, 0, probe.COORD_START + 10] == baseline[0, 0, probe.COORD_START + 11]
    assert probe.coordinate_summary(shadow['raw'], 10)['winner'] == 11
    assert probe.coordinate_summary(shadow['median'], 10)['winner'] == 11
    assert torch.equal(h, original) and torch.equal(baseline, observed)
    shifted = probe.HeadCapture(cell, 1, transform, 'fixture')
    shifted.hook(head, (h,), baseline)
    with pytest.raises(ValueError, match='shifted'):
        shifted.bind(torch.tensor([[42, 43]]), baseline[:, -1].float())
    transform.factors[0] += 1
    with pytest.raises(ValueError, match='factors changed'):
        cap.finish(transform, [probe.COORD_START + 10], [dict(coordinate_scores=normalized[0, probe.COORD_START:].tolist())])


def test_semantic_censoring_and_all_annotation_overlaps(packet):
    cell = packet['conditions'][0]
    session = FixtureSession(probe.PREVIOUS / 'checkpoint-16', packet)
    record = session.generate(cell)
    analysis = probe.analyze(record, cell, packet, session.tokenizer)
    row = analysis['selected_row']
    assert row['role'] == 'uncued' and not row['physical_recovery_credit']
    assert len(row['same_category_overlaps']) == sum(o['desc'] == 'bottle' for o in packet['images']['351017']['objects'])
    session.q.model.early_eos = True
    capped = session.generate(cell)
    probe.validate_record(capped, cell, packet)
    absent = probe.analyze(capped, cell, packet, session.tokenizer)
    assert absent['selected_row'] is None and absent['emitted_eos'] and absent['malformed']
    assert absent['alignment'] == {}
    changed = copy.deepcopy(record)
    changed['token_ids'][15] = 8987
    changed['text'] = session.tokenizer.decode(changed['token_ids'], skip_special_tokens=False)
    assert probe.analyze(changed, cell, packet, session.tokenizer)['selected_row'] is None
    wrong = dict(record, width=record['width'] + 1)
    with pytest.raises(ValueError, match='dimensions'):
        probe.analyze(wrong, cell, packet, session.tokenizer)


@pytest.mark.parametrize('mutation', ['budget', 'count', 'owner', 'prefix', 'force', 'context'])
def test_frozen_matrix_has_teeth(packet, mutation):
    altered = copy.deepcopy(packet)
    if mutation == 'budget':
        altered['conditions'][0]['budget'] += 1
    elif mutation == 'count':
        altered['conditions'].pop()
    elif mutation == 'owner':
        altered['conditions'][5]['target_box'][0] += 1
    elif mutation == 'prefix':
        altered['conditions'][0]['expected_ids'][0] += 1
    elif mutation == 'force':
        altered['conditions'][2]['forced_actions']['14'] += 1
    else:
        altered['requests']['13348']['prompt_token_ids'].append(0)
    with pytest.raises(ValueError):
        probe.validate_conditions(altered)


def test_same_image_fidelity_hold_and_saved_companion_corruption(prepared):
    directory, packet = prepared
    FixtureSession.break_image = 351017
    output = probe.OUTPUT / 'cpu-hold'
    try:
        assert probe.main(['run', '--config', str(directory / 'native-proposal.json'), '--output', str(output)], cpu_factory=FixtureSession) == 2
    finally:
        FixtureSession.break_image = None
    report = probe.a.load(output / 'readback.json')
    assert report['completed_requests'] == 4 and report['generated_actions'] == 727
    assert report['denominators']['observed'] == dict(x1_cued=1, uncued=2, teacher_path=1)
    assert sum(c['status'] == 'HOLD' for c in report['conditions']) == 6
    records = {c['condition']: probe.a.load(c['artifact']['path']) for c in report['conditions'] if c['status'] == 'completed'}
    original = (output / 'companion.json').read_text()
    for mutation in ('feedback', 'mass', 'factors', 'position'):
        changed = json.loads(original)
        if mutation == 'feedback':
            changed['feedback_into_generation'] = True
        elif mutation == 'mass':
            changed['full_vocabulary_scores'] = True
        elif mutation == 'factors':
            changed['projections'][0]['factors'][0] += 1
        else:
            changed['projections'][0]['position'] += 1
        (output / 'companion.json').write_text(json.dumps(changed))
        with pytest.raises(ValueError, match='companion'):
            probe.validate_companion(output, records, packet)
    (output / 'companion.json').write_text(original)
    with pytest.raises(ValueError, match='release'):
        probe.load_packet(directory / 'native-proposal.json')


def test_teacher_and_x1_supplied_accounting_and_companion_hold(packet):
    assert sum(c['budget'] for c in packet['conditions']) == 841
    assert sum(len(c['forced_actions']) for c in packet['conditions']) == 807
    session = FixtureSession(probe.PREVIOUS / 'checkpoint-16', packet)
    for index, supplied in [(8, [231]), (9, [231, 232, 233, 234])]:
        cell = packet['conditions'][index]
        record = session.generate(cell)
        probe.validate_record(record, cell, packet)
        row = probe.analyze(record, cell, packet, session.tokenizer)['selected_row']
        assert row['supplied_coordinate_actions'] == supplied and row['role'] == cell['population']
        assert row['physical_recovery_credit'] is False
        assert row['designated_annotation_id'] == 191150
        if index == 9:
            assert row['box'] == [546, 633, 556, 682]
            assert all(record['policy_logprobs'][i] == 0 for i in supplied)
            assert record['raw_steps'][231]['pre_force_logprob'] < -20
    cell = packet['conditions'][0]
    observed = session.generate(cell)
    session.companion_rows = None
    session.companion_status = dict(status='HOLD', reason='unsupported_fixture_composition', primary_untouched=True)
    held = session.generate(cell)
    assert held['companion_capture']['status'] == 'HOLD'
    assert all(observed[k] == held[k] for k in ('token_ids', 'raw_steps', 'median_steps', 'raw_logprobs', 'policy_logprobs'))
    assert not session.q.model.head._forward_hooks


def test_admission_source_self_inclusion_staleness_and_resources(prepared, monkeypatch):
    directory, packet = prepared
    proposal = directory / 'native-proposal.json'
    assert set(packet['producer']['files']) == set(probe.SOURCE_PATHS)
    config, _ = probe.load_packet(proposal, cpu=True)
    for mutation in ('producer', 'runtime', 'checkpoint'):
        altered = copy.deepcopy(config)
        if mutation == 'producer':
            altered['producer_files'][probe.SOURCE_PATHS[1]] = '0' * 64
        elif mutation == 'runtime':
            altered['runtime']['python'] = 'wrong'
        else:
            changed_packet = copy.deepcopy(packet)
            changed_packet['bindings']['checkpoint']['sha256'] = '0' * 64
            changed_path = directory / 'wrong-checkpoint-packet.json'
            probe.a.write(changed_path, changed_packet)
            altered['input_packet'] = probe.binding(changed_path)
        wrong = directory / (mutation + '.json')
        probe.a.write(wrong, altered)
        with pytest.raises(ValueError):
            probe.load_packet(wrong, cpu=True)
    monkeypatch.setattr(probe.readout, 'usage', lambda *args, **kwargs: dict(rss_bytes=probe.BOUNDS['rss_bytes'] + 1, retained_bytes=0))
    output = probe.OUTPUT / 'resource-hold'
    FixtureSession.loads = []
    assert probe.main(['run', '--config', str(proposal), '--output', str(output)], cpu_factory=FixtureSession) == 2
    assert FixtureSession.loads == []
    terminal = probe.a.load(output / 'terminal.json')
    assert terminal['counts']['attempted_requests'] == 0 and terminal['resource_excess'] == ['rss_bytes']
