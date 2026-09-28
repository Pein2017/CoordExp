import json

import pytest

from scripts import start_loss_benchmark as benchmark


@pytest.mark.parametrize('seed', [17, 29])
def test_continuation_runs_only_pending_arms_and_rejects_live_owner(tmp_path, monkeypatch, seed):
    monkeypatch.setattr(benchmark, 'ROOT', tmp_path)
    arms = list(benchmark.ARMS)
    if seed == 29:
        arms.reverse()
    names = [f'{arm}-order{seed}' for arm in arms]
    terminal = {'status': 'completed', 'completed_steps': 256, 'final_finite_status': 'finite'}
    benchmark.write_json(tmp_path / 'calibration.json', {})
    state_path = tmp_path / f'group-{seed}.json'
    benchmark.write_json(state_path, {'status': 'awaiting_authorization', 'active': None,
                                     'completed': names[:2], 'devices': '0,1,2,3'})
    for name in names[:2]:
        benchmark.write_json(tmp_path / 'train' / name / 'run.json', terminal)
        for step in (64, 256):
            benchmark.write_json(tmp_path / 'infer' / f'{name}-step{step}' / 'eval/metrics.json', {})
    launches, evaluations = [], []

    def command(argv, log, env):
        launches.append(log.stem)
        benchmark.write_json(tmp_path / 'train' / log.stem / 'run.json', terminal)

    monkeypatch.setattr(benchmark, 'command', command)
    monkeypatch.setattr(benchmark, 'prepare_cache', lambda *args: None)
    monkeypatch.setattr(benchmark, 'evaluate', lambda name, env: evaluations.append(name))
    benchmark.run_group(seed, '0,1,2,3', continue_existing=True)
    assert launches == names[2:]
    assert evaluations == [f'{name}-step{step}' for name in names[2:] for step in (64, 256)]
    state = json.loads(state_path.read_text())
    assert state['status'] == 'completed' and state['completed'] == names
    state.update(status='running', active=names[2])
    benchmark.write_json(state_path, state)
    with pytest.raises(RuntimeError, match='inactive authorization hold'):
        benchmark.run_group(seed, '0,1,2,3', continue_existing=True)
    assert launches == names[2:]
