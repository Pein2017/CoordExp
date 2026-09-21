"""One owner for the granted, finite two-arm history execution."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time

ROOT = Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-12-parallel-owner-research/history')
INPUT = ROOT / 'preparation/training-inputs.json'
PACKET = ROOT / 'preparation/packet-v2.json'
started = time.time()
receipts = []


def start(name, module, arguments, visible=None):
    path = ROOT / f'{name}.driver.log'
    stream = path.open('x')
    command = [sys.executable, '-m', module, *map(str, arguments)]
    env = dict(os.environ)
    if visible is not None:
        env['CUDA_VISIBLE_DEVICES'] = visible
    process = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
    entry = {'phase': name, 'pid': process.pid, 'command': command, 'visible_gpus': visible,
             'started': time.time(), 'log': str(path)}
    (ROOT / f'{name}.driver-launch.json').write_text(json.dumps(entry, indent=2) + '\n')
    print(json.dumps({'event': 'started', **entry}), flush=True)
    return process, stream, entry


def finish(job):
    process, stream, entry = job
    code = process.wait()
    stream.close()
    entry.update(exit_code=code, finished=time.time())
    receipts.append(entry)
    (ROOT / f"{entry['phase']}.driver-exit.json").write_text(json.dumps(entry, indent=2) + '\n')
    print(json.dumps({'event': 'finished', 'phase': entry['phase'], 'exit_code': code}), flush=True)
    return code


def run(name, module, arguments, visible=None):
    code = finish(start(name, module, arguments, visible))
    if code:
        raise RuntimeError(f'{name} exited {code}; see retained phase log')


try:
    jobs = []
    for arm, gpus in [('fixed_P', '2,3'), ('mixed_PQ', '0,1')]:
        jobs.append((arm, gpus, start('train-' + arm, 'probes.parallel_owner_research.training',
                     ['launch', '--input', INPUT, '--arm', arm, '--world-size', '2', '--output-root', ROOT / ('full-' + arm)], gpus)))
    failures = []
    for arm, gpus, job in jobs:
        if finish(job):
            failures.append(arm)
        else:
            run('cold-' + arm, 'probes.parallel_owner_research.training',
                ['cold-check', '--input', INPUT, '--arm', arm, '--output-root', ROOT / ('full-' + arm)], gpus.split(',')[0])
            if arm == 'mixed_PQ':
                print(json.dumps({'event': 'GPU_RELEASED', 'gpus': [0, 1], 'reason': 'mixed32 and coldcheck completed'}), flush=True)
    if failures:
        raise RuntimeError(f'Training failures retained: {failures}')
    for arm in ['Stable50', 'fixed_P', 'mixed_PQ']:
        endpoint = ROOT / 'preparation' / ('endpoint-' + arm + '.json')
        if arm != 'Stable50':
            run('endpoint-prepare-' + arm, 'probes.parallel_owner_research.history',
                ['endpoint-prepare', '--packet', PACKET, '--arm', arm, '--trained-root', ROOT / ('full-' + arm), '--endpoint-packet', endpoint])
        output = ROOT / ('endpoint-' + arm)
        run('endpoint-' + arm, 'probes.parallel_owner_research.history',
            ['endpoint-launch', '--endpoint-packet', endpoint, '--output', output])
        run('endpoint-merge-' + arm, 'probes.parallel_owner_research.history',
            ['endpoint-merge', '--endpoint-packet', endpoint, '--output', output])
    status = 'completed'
except BaseException as exc:
    status = 'failed'
    raise
finally:
    (ROOT / 'driver-receipt.json').write_text(json.dumps({'status': status, 'started': started,
             'finished': time.time(), 'phases': receipts}, indent=2) + '\n')
