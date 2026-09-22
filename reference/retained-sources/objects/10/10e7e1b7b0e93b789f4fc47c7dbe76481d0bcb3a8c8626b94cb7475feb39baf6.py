"""Replay candidate CPU/tensor checks without modifying sealed evidence."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import runpy
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
read = lambda p: json.loads(p.read_text())
binding = lambda p: dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest(), size_bytes=p.stat().st_size)
inventory = read(ROOT / 'artifact-bindings.json')
for record in inventory['records']:
    assert binding(Path(record['path'])) == record, record['path']
for pair in read(ROOT / 'source-snapshots.json'):
    for record in pair.values():
        assert binding(Path(record['path'])) == record, record['path']
terminal = read(ROOT / 'terminal.json')
for key in ('result', 'artifact_bindings', 'job_closure'):
    record = terminal[key]
    assert binding(Path(record['path'])) == record

jobs = [('independent_output_check.py', []), ('verify_stage1.py', ['SS','FF','SX_FY','FX_SY'])]
for path in sorted(ROOT.glob('state-verification-*.json')):
    jobs.append(('verify_state.py', [x['condition'] for x in read(path)['checks']]))
jobs += [('verify_rebuild.py', []), ('verify_components.py', [])]
reports = []
def verify_write(path, data, *args, **kwargs):
    assert path.exists(), str(path)
    assert json.loads(data) == read(path), str(path)
    reports.append(binding(path))
    return len(data)

for script, args in jobs:
    sys.argv = [str(ROOT / script), *args]
    with patch.object(Path, 'write_text', verify_write), contextlib.redirect_stdout(io.StringIO()):
        runpy.run_path(str(ROOT / script), run_name='__main__')
    print('passed', script, len(args), flush=True)

cost = read(ROOT / 'cost.json')
assert len(cost['receipts']) == 22
assert sum(x['model_forwards'] for x in cost['receipts']) == cost['model_forwards'] == 39368
assert all(int(p.read_text()) == 0 for p in (ROOT / 'logs').glob('*.exit'))
live = []
for record in cost['receipts']:
    proc = Path('/proc') / str(record['pid']) / 'cmdline'
    if proc.exists():
        live.append(dict(pid=record['pid'], argv=proc.read_bytes().decode(errors='replace').split('\0')))
assert not live, live
for record in inventory['records']:
    assert binding(Path(record['path'])) == record, record['path']
out = dict(status='passed', bound_files=len(inventory['records']), candidate_unchanged=True,
           saved_checks=reports, model_forwards=cost['model_forwards'], model_jobs=22,
           live_owned_jobs=live, new_model_forwards=0, tensor_checks_device='CPU',
           terminal=binding(ROOT / 'terminal.json'), script=binding(Path(__file__)))
(ROOT / 'coordination/lead-verification.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k not in ('saved_checks','terminal','script')}))
