"""One bounded eight-GPU launch after the lead's real smoke acceptance."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).parent
REPO = Path('/data/CoordExp/.worktrees/research-probes')


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


if __name__ == '__main__':
    packet_sha = hashlib.sha256((ROOT/'packet.json').read_bytes()).hexdigest()
    gate = json.loads((ROOT/'smoke-lead-acceptance.json').read_text())
    assert gate['status'] == 'lead-accepted' and gate['packet_sha256'] == packet_sha
    started = time.monotonic()
    full = ROOT/'full'; full.mkdir(exist_ok=False)
    jobs = []
    for rank in range(8):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(rank), OMP_NUM_THREADS='1')
        command = ['python', str(ROOT/'run_probe.py'), '--packet', str(ROOT/'packet.json'),
                   '--out-dir', str(full/f'rank-{rank}'), '--rank', str(rank)]
        log = (full/f'rank-{rank}.log').open('x')
        process = subprocess.Popen(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
        jobs.append((rank, process, log, command))
    write(full/'process-launch.json', dict(pid=os.getpid(), packet_sha256=packet_sha,
          jobs=[dict(rank=r, pid=p.pid, command=c) for r,p,_,c in jobs]))
    exits = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(p.wait):(r,p,log) for r,p,log,_ in jobs}
        for future in concurrent.futures.as_completed(futures):
            rank, process, log = futures[future]; code=future.result(); log.close()
            exits.append(dict(rank=rank, pid=process.pid, exit_code=code))
    write(full/'process-exits.json', sorted(exits,key=lambda x:x['rank']))
    success = all(x['exit_code'] == 0 for x in exits)
    write(full/'terminal.json', dict(status='completed' if success else 'failed',
        packet_sha256=packet_sha, wall_seconds=time.monotonic()-started, process_exits=exits))
    print('PANEL COMPLETED' if success else 'PANEL FAILED', flush=True)
    raise SystemExit(0 if success else 1)
