"""Event-driven observation of this pilot's first two steps or failure."""
import ctypes
import json
import os
from pathlib import Path
import select
import sys
import time

p=Path(__file__).parent
lib=ctypes.CDLL(None,use_errno=True)
fd=lib.inotify_init1(os.O_CLOEXEC)
assert fd>=0
watched=set()


def watch(path):
    if path.is_dir() and str(path) not in watched:
        assert lib.inotify_add_watch(fd,os.fsencode(path),0x8|0x100|0x80)>=0
        watched.add(str(path))


watch(p)
terminal_only=len(sys.argv)>1 and sys.argv[1]=='terminal'
deadline_seconds=10800 if terminal_only else 900
deadline=time.monotonic()+deadline_seconds
try:
    while True:
        watch(p/'training')
        found=False
        candidates=[('pipeline_terminal',p/'pipeline-terminal.json')]
        if not terminal_only:candidates.insert(0,('two_step_gate',p/'training/two-step-smoke.json'))
        for event,path in candidates:
            if path.is_file():
                try: value=json.loads(path.read_text())
                except json.JSONDecodeError: continue
                print(json.dumps(dict(event=event,receipt=value)),flush=True)
                found=True
                break
        if found:break
        remaining=deadline-time.monotonic()
        if remaining<=0:
            print(json.dumps(dict(event='observation_deadline',seconds=deadline_seconds)),flush=True)
            break
        if select.select([fd],[],[],remaining)[0]:os.read(fd,65536)
finally:
    os.close(fd)
