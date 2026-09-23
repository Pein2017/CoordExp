"""Account and stop this pilot's owned GPU processes under its shared ceiling."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-coordinate-codebook-alignment')
WALL_SECONDS = 8 * 3600
GPU_SECONDS = 64 * 3600
RESERVE_SECONDS = 120


@contextmanager
def ledger(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'cost.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = root / 'cost.json'
        data = json.loads(path.read_text()) if path.exists() else {'jobs': []}
        yield data
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
        os.replace(tmp, path)


def allocated_seconds(data, now):
    return sum(len(j['gpus']) * (j.get('terminal_time', now) - j['start_time'])
               for j in data['jobs'])


def run(name: str, gpus: list[int], command: list[str], root: Path = ROOT,
        *, cache_root: Path | None = None, reserve_seconds: int = RESERVE_SECONDS,
        wall_seconds: int = WALL_SECONDS, gpu_seconds: int = GPU_SECONDS) -> int:
    if not command or not gpus or len(gpus) != len(set(gpus)) or any(g not in range(8) for g in gpus):
        raise ValueError('require command and unique GPU IDs in 0..7')
    if not 60 <= reserve_seconds < wall_seconds or gpu_seconds <= 8 * reserve_seconds:
        raise ValueError('reserve must leave time to terminate and join producers')
    now = time.time()
    with ledger(root) as data:
        if any(j['name'] == name for j in data['jobs']):
            raise ValueError('attempt name already exists; preserve failures and use a fresh name')
        busy = {g for j in data['jobs'] if 'terminal_time' not in j for g in j['gpus']}
        if busy.intersection(gpus):
            raise ValueError('GPU already allocated to another owned live producer')
        limits = {'wall_seconds': wall_seconds, 'gpu_seconds': gpu_seconds}
        if data.setdefault('limits', limits) != limits:
            raise ValueError('cost ledger limits differ from this launch')
        start = data.setdefault('wall_start', now)
        data['clock_boundary'] = 'first model-entry process launch, conservatively includes loading/startup'
        if now >= start + wall_seconds - reserve_seconds or allocated_seconds(data, now) >= gpu_seconds - 8 * reserve_seconds:
            raise RuntimeError('pilot budget reserve reached')
        job = {'name': name, 'gpus': gpus, 'command': command, 'start_time': now,
               'supervisor_pid': os.getpid(), 'state': 'launching',
               'stop_reserve_seconds': reserve_seconds,
               'outstanding_reservation_gpu_seconds': len(gpus) * reserve_seconds}
        data['jobs'].append(job)
    logs = root / 'execution'
    logs.mkdir(exist_ok=True)
    proc = None
    reason = 'completed'
    code = 1
    try:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(map(str, gpus)), PYTHONUNBUFFERED='1')
        env['coordexp_infras_PACK_CACHE_ROOT'] = str(cache_root or root / 'packing-cache')
        env.setdefault('OMP_NUM_THREADS', '4')
        with (logs / f'{name}.log').open('xb') as log:
            proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env,
                                    start_new_session=True)
            with ledger(root) as data:
                j = next(j for j in data['jobs'] if j['name'] == name)
                j.update(pid=proc.pid, state='running', log=str(logs / f'{name}.log'))
            while True:
                try:
                    code = proc.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    with ledger(root) as data:
                        now = time.time()
                        stop = (now >= data['wall_start'] + wall_seconds - reserve_seconds
                                or allocated_seconds(data, now) >= gpu_seconds - 8 * reserve_seconds)
                    if stop:
                        reason = 'budget_reserve'
                        os.killpg(proc.pid, signal.SIGTERM)
                        try:
                            code = proc.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            os.killpg(proc.pid, signal.SIGKILL)
                            code = proc.wait()
                        break
    except BaseException:
        reason = 'supervisor_failure_or_interrupt'
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        raise
    finally:
        with ledger(root) as data:
            j = next(j for j in data['jobs'] if j['name'] == name)
            j.update(terminal_time=time.time(), state='terminal', exit_code=code, reason=reason)
            data['allocated_gpu_seconds'] = allocated_seconds(data, time.time())
        print(json.dumps(j), flush=True)
    return code


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--gpus', required=True)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--cache-root', type=Path)
    parser.add_argument('--reserve-seconds', type=int, default=RESERVE_SECONDS)
    parser.add_argument('--wall-seconds', type=int, default=WALL_SECONDS)
    parser.add_argument('--gpu-seconds', type=int, default=GPU_SECONDS)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    raise SystemExit(run(args.name, [int(x) for x in args.gpus.split(',')], command,
                         args.root, cache_root=args.cache_root,
                         reserve_seconds=args.reserve_seconds,
                         wall_seconds=args.wall_seconds, gpu_seconds=args.gpu_seconds))


if __name__ == '__main__':
    main()
