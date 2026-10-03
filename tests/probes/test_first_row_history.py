"""CPU-only causal forcing, fail-closed ordering and real CLI/publication checks."""
from pathlib import Path
from types import SimpleNamespace
import json
import os
import subprocess
import sys

import pytest
import torch

from probes import first_row_history as probe, rollout_row_credit as retained
from src.qwen.coordinate_policy import MedianPolicy


@pytest.fixture(scope='module')
def prepared(tmp_path_factory):
    path = os.environ.get('FIRST_ROW_HISTORY_PREPARED')
    if path:
        return Path(path)
    path = tmp_path_factory.mktemp('history') / 'prepared'
    assert probe.main(['prepare', '--output', str(path)]) == 0
    return path


@pytest.fixture(scope='module')
def frontend():
    q = retained.frontend()
    assert q.model is None
    return q


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


class SyntheticCachedModel:
    """No nn.Module or forward: fake only the unavailable model computation."""
    def __init__(self, q, packet, version):
        self.tokenizer = q.tokenizer
        self.native = packet['selected'][str(version)]['expected_ids']
        _, sequence, _ = probe.data.full_label_sequence(packet['image'], q)
        self.teacher = list(sequence.input_ids[len(packet['request']['prompt_token_ids']):])
        coords = [q.tokenizer.convert_tokens_to_ids(f'<|coord_{i}|>') for i in range(1000)]
        head = SimpleNamespace(weight=torch.ones(len(q.tokenizer), 1, dtype=torch.bfloat16), bias=None,
            selected_token_ids=torch.tensor(coords), shared_embed_delta=torch.zeros(1000, 1))
        self.norm = MedianPolicy(SimpleNamespace(get_output_embeddings=lambda: head,
            get_input_embeddings=lambda: SimpleNamespace(shared_embed_delta=torch.zeros(1000, 1))), coords)
        self.captured = []

    def generate(self, **kwargs):
        assert kwargs['generation_config'].do_sample is False
        assert kwargs['max_new_tokens'] == 73 and kwargs['repetition_penalty'] == 1
        assert kwargs['output_logits'] and kwargs['output_scores']
        prompt_width = kwargs['input_ids'].shape[1]
        ids = kwargs['input_ids'].clone()
        self.captured.append(ids.tolist())
        raw, processed = [], []
        for step in range(kwargs['max_new_tokens']):
            history = ids[0, prompt_width:].tolist()
            # Depend on the actual emitted history, not the requested condition label.
            teacher_history = step >= 9 and history[:9] == probe.PREFIXES['GT']
            token = self.teacher[step] if teacher_history else self.native[step]
            logits = torch.zeros(1, len(self.tokenizer))
            logits[0, token] = 8
            raw.append(logits)
            scores = kwargs['logits_processor'](ids, logits)
            processed.append(scores)
            selected = scores.argmax(-1).reshape(1, 1)
            ids = torch.cat((ids, selected), dim=1)
            if int(selected) == kwargs['eos_token_id']:
                break
        return SimpleNamespace(sequences=ids, logits=tuple(raw), scores=tuple(processed))


def fixture_factory(frontend):
    class FixtureSession(probe.NativeSession):
        def __init__(self, checkpoint, packet):
            version = int(checkpoint.name.split('-')[1])
            model = SyntheticCachedModel(frontend, packet, version)
            self.q = SimpleNamespace(model=model, tokenizer=frontend.tokenizer)
            self.batch = probe.verify_frontend(frontend, packet['image'], packet['request'])
            self.norm = model.norm
            self.composition = dict(kind='CPU_FIXTURE', model_loaded=False, model_forwards=0,
                                    substitution='synthetic singleton cached decision/logit computation only')

        def generate(self, history):
            # Same production method, replacing CUDA autocast only; no real model.
            original = torch.autocast
            torch.autocast = lambda *args, **kwargs: original('cpu', enabled=False)
            try:
                return super().generate(history)
            finally:
                torch.autocast = original

        def close(self):
            self.q.model = None
    return FixtureSession


def test_real_cli_fake_compute_publication_and_fresh_consumer(prepared, frontend, tmp_path):
    output = Path(os.environ.get('FIRST_ROW_HISTORY_CPU_OUTPUT', tmp_path / 'cpu-qualification'))
    config = prepared / 'native-proposal.json'
    code = probe.main(['run', '--config', str(config), '--output', str(output)], cpu_factory=fixture_factory(frontend))
    terminal = probe.a.load(output / 'terminal.json')
    assert code == 0, probe.a.load(output / 'error.json') if (output / 'error.json').exists() else terminal
    assert terminal['counts'] == dict(checkpoint_loads=0, fixture_sessions=2, attempted_requests=4,
        completed_requests=4, generated_actions=292, optimizer=0, backward=0, replay=0, exports=0)
    readback = probe.a.load(output / 'readback.json')
    assert readback['compute'] == 'CPU_FIXTURE' and readback['complete']
    assert [row['condition'] for row in readback['conditions']] == probe.CONDITIONS
    assert all(row['natural_fidelity']['qualified'] for row in readback['conditions'][::2])
    assert readback['conditions'][1]['analysis']['first_bottle']['target_bottle_iou'] == 1
    fresh = subprocess.run([sys.executable, '-m', 'probes.first_row_history', 'readback', '--output', str(output)],
        cwd=probe.ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2'), capture_output=True, text=True)
    assert fresh.returncode == 0, fresh.stderr
    assert '"completed_requests": 4' in fresh.stdout and '"generated_actions": 292' in fresh.stdout
    with pytest.raises(FileExistsError):
        probe.main(['run', '--config', str(config), '--output', str(output)], cpu_factory=fixture_factory(frontend))


def test_forcing_keeps_raw_scores_and_stops_after_nine():
    forcer = probe.PrefixForcer(probe.PREFIXES['GT'], 3)
    scores = torch.arange(152670, dtype=torch.float32).reshape(1, -1) / 100000
    original = scores.clone()
    for step in range(9):
        forced = forcer(torch.zeros(1, 3 + step, dtype=torch.long), scores)
        assert forced is not scores and forced.data_ptr() != scores.data_ptr()
        assert torch.equal(scores, original)
        assert int(forced.argmax()) == probe.PREFIXES['GT'][step]
        observation = forcer.observations[step]
        assert observation['pre_force_argmax'] == 152669
        assert observation['unforced_median_logprob'] == float(torch.log_softmax(original, -1)[0, probe.PREFIXES['GT'][step]])
    for step in range(9, 73):
        assert forcer(torch.zeros(1, 3 + step, dtype=torch.long), scores) is scores
    assert len(forcer.observations) == 9 and forcer.calls == 73
    with pytest.raises(ValueError, match='sequential'):
        forcer(torch.zeros(1, 3 + 75, dtype=torch.long), scores)


def test_actual_cached_prefix_changes_free_decision_and_likelihood_channels(prepared, frontend):
    packet = probe.a.load(prepared / 'input-packet.json')
    session = fixture_factory(frontend)(Path(packet['selected']['0']['checkpoint']), packet)
    native, gt = session.generate('native'), session.generate('GT')
    assert native['token_ids'] == packet['selected']['0']['expected_ids']
    assert gt['token_ids'][:9] == probe.PREFIXES['GT']
    assert gt['token_ids'][9:] == session.q.model.teacher[9:73]
    assert session.q.model.captured == [[packet['request']['prompt_token_ids']]] * 2  # Never a9-token prefill.
    assert [row['position'] for row in gt['prefix_observations'] if row['pre_force_argmax'] != row['requested_token_id']] == [5, 6, 7]
    assert gt['policy_logprobs'][:9] == [0.] * 9
    assert all(value < 0 for value in gt['raw_logprobs'][:9])
    assert [row['unforced_median_logprob'] for row in gt['prefix_observations']] == gt['raw_logprobs'][:9]
    assert gt['raw_logprobs'][9:] == gt['policy_logprobs'][9:]
    assert probe.analyze(native | dict(request_id='native', width=packet['image']['width'], height=packet['image']['height']),
                         packet['image'], frontend.tokenizer)['first_free_category'] == 'person'
    assert probe.analyze(gt | dict(request_id='GT', width=packet['image']['width'], height=packet['image']['height']),
                         packet['image'], frontend.tokenizer)['first_free_category'] == 'bottle'


def test_pre_force_observation_is_after_real_median_normalization(prepared, frontend):
    packet = probe.a.load(prepared / 'input-packet.json')
    session = fixture_factory(frontend)(Path(packet['selected']['0']['checkpoint']), packet)
    session.norm.head.weight[probe.PREFIXES['native'][5]] = 2
    session.norm.head.weight[probe.PREFIXES['GT'][5]] = .5
    record = session.generate('GT')
    raw = torch.zeros(1, len(frontend.tokenizer))
    raw[0, probe.PREFIXES['native'][5]] = 8
    median = session.norm.generation_transform()(torch.tensor([[0]]), raw)
    expected = float(torch.log_softmax(median.float(), -1)[0, probe.PREFIXES['GT'][5]])
    assert record['prefix_observations'][5]['unforced_median_logprob'] == expected
    assert record['raw_logprobs'][5] == float(torch.log_softmax(raw, -1)[0, probe.PREFIXES['GT'][5]])
    assert record['raw_logprobs'][5] != expected and record['policy_logprobs'][5] == 0


@pytest.mark.parametrize('version, failure', [(0, 'argmax'), (0, 'ids'), (16, 'argmax'), (16, 'ids')])
def test_natural_hold_blocks_dependent_gt_and_remaining_requests(prepared, frontend, tmp_path, version, failure):
    factory = fixture_factory(frontend)
    called = []
    class CorruptSession(factory):
        def __init__(self, checkpoint, packet):
            super().__init__(checkpoint, packet)
            self.version = int(checkpoint.name.split('-')[1])
        def generate(self, history):
            called.append(f'A{self.version}/{history}')
            record = super().generate(history)
            if self.version == version and history == 'native':
                if failure == 'argmax':
                    record['prefix_observations'][5]['pre_force_argmax'] = probe.PREFIXES['GT'][5]
                else:
                    record['token_ids'][20] = probe.PREFIXES['GT'][5]
                    record['text'] = self.q.tokenizer.decode(record['token_ids'], skip_special_tokens=False)
            return record
    output = tmp_path / f'hold-{version}-{failure}'
    assert probe.main(['run', '--config', str(prepared / 'native-proposal.json'), '--output', str(output)], cpu_factory=CorruptSession) == 2
    assert called == probe.CONDITIONS[:1 if version == 0 else 3]
    terminal = probe.a.load(output / 'terminal.json')
    assert terminal['status'] == 'technical_HOLD' and not (output / 'error.json').exists()
    readback = probe.a.load(output / 'readback.json')
    assert readback['partial'] and not readback['complete']
    assert readback['conditions'][-1]['natural_fidelity']['qualified'] is False
    assert not (output / f'A{version}-GT.json').exists()
    if version == 0:
        assert not (output / 'A16-native.json').exists()
    else:
        assert (output / 'A0-GT.json').exists()
    assert probe.main(['readback', '--output', str(output)]) == 2


def test_compute_exception_is_terminal_and_preserves_completed_evidence(prepared, frontend, tmp_path):
    factory = fixture_factory(frontend)
    class FailSession(factory):
        def generate(self, history):
            if history == 'GT':
                raise RuntimeError('fixture unavailable compute failure')
            return super().generate(history)
    output = tmp_path / 'failure'
    assert probe.main(['run', '--config', str(prepared / 'native-proposal.json'), '--output', str(output)], cpu_factory=FailSession) == 1
    assert probe.a.load(output / 'terminal.json')['status'] == 'failure'
    assert probe.a.load(output / 'terminal.json')['counts']['attempted_requests'] == 2
    assert probe.a.load(output / 'terminal.json')['counts']['completed_requests'] == 1
    assert 'unavailable compute failure' in probe.a.load(output / 'error.json')['message']
    assert (output / 'A0-native.json').exists() and not (output / 'A0-GT.json').exists()
    assert not (output / 'load-A16.json').exists()


def test_native_unreleased_fails_before_any_checkpoint_or_device_load(prepared, tmp_path, monkeypatch):
    monkeypatch.setattr(probe, 'NativeSession', lambda *_: pytest.fail('unreleased model load'))
    monkeypatch.setattr(probe, 'qualify_payload', lambda *_: pytest.fail('unreleased payload load'))
    output = tmp_path / 'unreleased'
    assert probe.main(['run', '--config', str(prepared / 'native-proposal.json'), '--output', str(output)]) == 1
    terminal = probe.a.load(output / 'terminal.json')
    assert terminal['counts']['checkpoint_loads'] == terminal['counts']['attempted_requests'] == 0
    assert 'exact clean lead release' in probe.a.load(output / 'error.json')['message']
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('mutation', ['budget', 'target', 'prefix', 'saved_ids'])
def test_packet_drift_fails_closed_before_compute(prepared, tmp_path, mutation):
    config = probe.a.load(prepared / 'native-proposal.json')
    packet = probe.a.load(prepared / 'input-packet.json')
    if mutation == 'budget':
        config['bounds']['actions'] = 293
    elif mutation == 'target':
        packet['target']['bbox_2d'][0] += 1
    elif mutation == 'prefix':
        packet['prefixes']['GT'][5] = probe.PREFIXES['native'][5]
    else:
        packet['selected']['0']['expected_ids'][30] += 1
    path = tmp_path / 'input.json'
    probe.a.write(path, packet)
    config['input_packet'] = probe.small_binding(path)
    path = tmp_path / 'config.json'
    probe.a.write(path, config)
    output = tmp_path / 'drift'
    assert probe.main(['run', '--config', str(path), '--output', str(output)], cpu_factory=lambda *_: pytest.fail('drift reached compute')) == 1
    assert probe.a.load(output / 'terminal.json')['counts']['attempted_requests'] == 0
    assert (output / 'error.json').exists()


def rendered_record(frontend, packet, text):
    tokens = frontend.tokenizer(text, add_special_tokens=False)['input_ids']
    return dict(request_id='accounting', image_id=probe.IMAGE_ID, width=packet['image']['width'],
        height=packet['image']['height'], text=text, token_ids=tokens, stop_reason='length')


def test_free_target_accounting_duplicates_invalidity_and_boundary_rows(prepared, frontend):
    packet = probe.a.load(prepared / 'input-packet.json')
    def row(desc, box):
        return '<|object_ref_start|>' + desc + '<|object_ref_end|><|box_start|>' + ''.join(f'<|coord_{x}|>' for x in box) + '<|box_end|>'
    forced = frontend.tokenizer.decode(probe.PREFIXES['native'], skip_special_tokens=False)
    record = rendered_record(frontend, packet, forced + row('person', [0, 13, 536, 990]) +
        row('bottle', probe.TARGET_BOX) + row('bottle', [300, 100, 200, 110]) + '<|object_ref_start|>bottle<|object_ref_end|><|box_start|>')
    result = probe.analyze(record, packet['image'], frontend.tokenizer)
    assert len(result['free_rows']) == 3 and result['forced_row']['description'] == 'person'
    assert result['first_free_category'] == 'person'
    assert result['first_free_row']['target_localized'] is False
    assert result['first_bottle']['target_localized'] and result['first_bottle']['target_bottle_iou'] == 1
    assert result['any_free_target_localized'] and result['invalid_free_rows'] == 1
    assert result['duplicate_events_including_forced'][0]['partner_orders'] == [0]
    assert result['duplicate_events_including_forced'][0]['order'] == 1
    assert result['malformed'][-1]['censored']
    assert result['free_rows'][1]['positions'][0] == 18
    assert result['free_rows'][1]['same_category_owner_candidates'][0]['coco_ann_id'] == probe.TARGET_ID
    # The supplied person alone gives no free coverage or target recovery.
    alone = probe.analyze(rendered_record(frontend, packet, forced), packet['image'], frontend.tokenizer)
    assert alone['free_rows'] == [] and not alone['any_free_target_localized']


@pytest.mark.parametrize('channel', ['raw_logprobs', 'policy_logprobs', 'prefix_observations'])
def test_nonfinite_trace_never_passes_validation(prepared, frontend, channel):
    packet = probe.a.load(prepared / 'input-packet.json')
    record = fixture_factory(frontend)(Path(packet['selected']['0']['checkpoint']), packet).generate('native')
    if channel == 'prefix_observations':
        record[channel][0]['unforced_median_logprob'] = float('nan')
    else:
        record[channel][0] = float('nan')
    with pytest.raises(ValueError, match='nonfinite|incomplete'):
        probe.validate_record(record, 'native')


def test_consumer_rejects_conditions_published_after_hold(prepared, frontend, tmp_path):
    packet = probe.a.load(prepared / 'input-packet.json')
    session = fixture_factory(frontend)(Path(packet['selected']['0']['checkpoint']), packet)
    record = session.generate('native')
    record['prefix_observations'][0]['pre_force_argmax'] = 1
    record.update(request_id='hold', width=packet['image']['width'], height=packet['image']['height'])
    probe.a.write(tmp_path / 'A0-native.json', record)
    probe.a.write(tmp_path / 'A0-GT.json', record)
    with pytest.raises(ValueError, match='HOLD ordering'):
        probe.consume(tmp_path, packet, frontend.tokenizer)


def test_consumer_binds_likelihood_bytes_beyond_text_and_geometry(prepared, frontend, tmp_path):
    packet = probe.a.load(prepared / 'input-packet.json')
    record = fixture_factory(frontend)(Path(packet['selected']['0']['checkpoint']), packet).generate('native')
    record.update(request_id='trace-integrity', width=packet['image']['width'], height=packet['image']['height'])
    path = tmp_path / 'A0-native.json'
    probe.a.write(path, record)
    original = probe.consume(tmp_path, packet, frontend.tokenizer)['conditions'][0]
    record['raw_logprobs'][5] -= .125
    path.write_text(json.dumps(record, allow_nan=False))
    changed = probe.consume(tmp_path, packet, frontend.tokenizer)['conditions'][0]
    assert original['analysis'] == changed['analysis'] and original['token_ids'] == changed['token_ids']
    assert original['artifact']['sha256'] != changed['artifact']['sha256']
