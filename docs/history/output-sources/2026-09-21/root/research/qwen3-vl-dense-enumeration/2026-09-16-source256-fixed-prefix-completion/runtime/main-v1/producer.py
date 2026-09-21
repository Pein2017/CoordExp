from __future__ import annotations
import hashlib, json, os, pathlib, subprocess, sys, time, traceback

ROOT = pathlib.Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-16-source256-fixed-prefix-completion/runtime/main-v1')
WORKTREE = pathlib.Path('/data/CoordExp/.worktrees/research-probes')
PACKET = ROOT / 'evaluation/packet.json'
RELEASE = pathlib.Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-16-source256-fixed-prefix-completion/lead/main-release.json')
HANDOFF = ROOT / 'handoff.log'
EVENTS = ROOT / 'events.jsonl'


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False) + '\n').encode()


def binding(path):
    path = pathlib.Path(path).resolve(strict=True); data = path.read_bytes()
    return {'path': str(path), 'sha256': hashlib.sha256(data).hexdigest(), 'size_bytes': len(data)}


def publish(path, value):
    path = pathlib.Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(canonical(value)); stream.flush(); os.fsync(stream.fileno())


def event(kind, **fields):
    value = {'event': kind, 'unix_time': time.time(), **fields}
    with EVENTS.open('ab') as stream:
        stream.write(canonical(value)); stream.flush(); os.fsync(stream.fileno())


def handoff(line):
    with HANDOFF.open('a', encoding='utf-8') as stream:
        stream.write(line + '\n'); stream.flush(); os.fsync(stream.fileno())


def spawn(command, *, devices, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    handle = log.open('xb')
    env = os.environ.copy(); env['CUDA_VISIBLE_DEVICES'] = devices
    process = subprocess.Popen(command, cwd=WORKTREE, env=env, stdout=handle, stderr=subprocess.STDOUT)
    return process, handle


def wait_group(items):
    exits = []
    for item in items:
        code = item['process'].wait(); item['handle'].close()
        exits.append({key: value for key, value in item.items() if key not in {'process', 'handle'}} | {'exit_code': code})
    return exits


def main():
    started = time.time(); packet = json.loads(PACKET.read_text()); release = json.loads(RELEASE.read_text())
    assert release['schema'] == 'source256.lead_main_release.v1' and release['status'] == 'lead-accepted'
    assert release['preparation']['sha256'] == packet['sources']['preparation']['sha256']
    assert packet['scarcity_gate']['status'] == 'passed' and packet['bounds']['training_updates_per_arm'] == 64
    assert not (ROOT / 'launch.json').exists() and not EVENTS.exists()
    event('producer_started', pid=os.getpid(), packet=binding(PACKET), release=binding(RELEASE))
    running = []
    for spec in packet['training_launch']:
        log = ROOT / 'logs' / f"training-{spec['arm']}.log"
        process, handle = spawn(spec['command'], devices=','.join(map(str, spec['visible_devices'])), log=log)
        running.append({'arm': spec['arm'], 'pid': process.pid, 'visible_devices': spec['visible_devices'], 'command': spec['command'], 'log': str(log), 'process': process, 'handle': handle})
    publish(ROOT / 'launch.json', {'schema': 'source256.main_v1.launch.v1', 'status': 'spawned', 'packet': binding(PACKET), 'release': binding(RELEASE), 'producer': binding(__file__), 'controller_pid': os.getpid(), 'arms': [{key: value for key, value in item.items() if key not in {'process', 'handle'}} for item in running]})
    event('training_launched', arms=[{'arm': item['arm'], 'pid': item['pid']} for item in running])
    exits = wait_group(running); publish(ROOT / 'training-exits.json', {'schema': 'source256.main_v1.training_exits.v1', 'arms': exits})
    if any(item['exit_code'] != 0 for item in exits):
        raise RuntimeError(f'training failed: {exits}')
    terminals = {}
    for spec in packet['training_launch']:
        path = pathlib.Path(spec['terminal']).resolve(strict=True); value = json.loads(path.read_text())
        assert value['status'] == 'completed' and value['updates'] == 64 and value['optimizer_mode'] == 'fresh'
        terminals[spec['arm']] = binding(path)
    event('training_complete', terminals=terminals)
    handoff('SOURCE256_TRAINING_COMPLETE ' + json.dumps(terminals, sort_keys=True))

    readback_exits = []
    for endpoint in packet['endpoints']:
        for split in ('train', 'dev'):
            jobs = sorted((job for job in endpoint['jobs'] if job['split'] == split), key=lambda job: job['shard'])
            assert len(jobs) == 8
            running = []
            for job in jobs:
                output = pathlib.Path(job['output']); assert not output.exists()
                log = ROOT / 'logs/readback' / endpoint['label'] / split / f"shard-{job['shard']:02d}.log"
                process, handle = spawn(job['command'], devices=str(job['visible_device']), log=log)
                running.append({'endpoint': endpoint['label'], 'split': split, 'shard': job['shard'], 'pid': process.pid, 'visible_device': job['visible_device'], 'command': job['command'], 'log': str(log), 'output': str(output), 'process': process, 'handle': handle})
            event('readback_wave_launched', endpoint=endpoint['label'], split=split, jobs=[{'shard': item['shard'], 'pid': item['pid']} for item in running])
            exits = wait_group(running); readback_exits.extend(exits)
            if any(item['exit_code'] != 0 for item in exits):
                publish(ROOT / 'readback-exits.json', {'schema': 'source256.main_v1.readback_exits.v1', 'status': 'failed', 'jobs': readback_exits})
                raise RuntimeError(f"readback failed: {endpoint['label']} {split}: {exits}")
            for item in exits:
                value = json.loads(pathlib.Path(item['output']).read_text()); assert value['status'] == 'completed_unscored' and value['batch_size'] == 4
            event('readback_wave_complete', endpoint=endpoint['label'], split=split)
    publish(ROOT / 'readback-exits.json', {'schema': 'source256.main_v1.readback_exits.v1', 'status': 'completed', 'jobs': readback_exits})

    reducer = packet['reducer']; reducer_log = ROOT / 'logs/reducer.log'; reducer_log.parent.mkdir(parents=True, exist_ok=True)
    with reducer_log.open('xb') as stream:
        code = subprocess.run(reducer['command'], cwd=WORKTREE, stdout=stream, stderr=subprocess.STDOUT).returncode
        stream.flush(); os.fsync(stream.fileno())
    if code != 0:
        raise RuntimeError(f'reducer failed with exit {code}')
    result_path = pathlib.Path(reducer['output']).resolve(strict=True); result = json.loads(result_path.read_text())
    assert result['status'] == 'completed_saved_readback_evaluation'
    event('reducer_complete', result=binding(result_path))
    terminal = {'schema': 'source256.main_v1.runtime_terminal.v1', 'status': 'completed', 'elapsed_seconds': time.time() - started, 'packet': binding(PACKET), 'release': binding(RELEASE), 'training_terminals': terminals, 'training_exits': binding(ROOT / 'training-exits.json'), 'readback_exits': binding(ROOT / 'readback-exits.json'), 'result': binding(result_path), 'events': binding(EVENTS)}
    publish(ROOT / 'runtime-terminal.json', terminal)
    handoff('SOURCE256_RUNTIME_COMPLETE ' + json.dumps(binding(ROOT / 'runtime-terminal.json'), sort_keys=True))


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        failure = {'schema': 'source256.main_v1.runtime_failure.v1', 'status': 'failed', 'unix_time': time.time(), 'error_type': type(exc).__name__, 'error': str(exc), 'traceback': traceback.format_exc(), 'packet': binding(PACKET), 'release': binding(RELEASE)}
        failure_path = ROOT / 'runtime-failure.json'
        if not failure_path.exists(): publish(failure_path, failure)
        try: event('runtime_failed', failure=binding(failure_path))
        except Exception: pass
        handoff('SOURCE256_RUNTIME_FAILED ' + json.dumps(binding(failure_path), sort_keys=True))
        raise
