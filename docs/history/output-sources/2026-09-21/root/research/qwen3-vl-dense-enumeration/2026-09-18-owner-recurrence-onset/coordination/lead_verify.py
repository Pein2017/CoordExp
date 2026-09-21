"""Replay the frozen CPU consumers without overwriting candidate artifacts."""
import hashlib
import importlib.util
import json
from pathlib import Path
import runpy
import sys
from unittest.mock import patch

import torch

R = Path(__file__).resolve().parent.parent
sys.path.insert(0, '/data/CoordExp/.worktrees/research-probes')
torch.set_num_threads(4)
read = lambda p: json.loads(Path(p).read_text())
bindings = read(R / 'artifact-bindings.json')['files']
for b in bindings:
    p = Path(b['path'])
    assert p.stat().st_size == b['size_bytes'], p
    assert hashlib.sha256(p.read_bytes()).hexdigest() == b['sha256'], p
spec = importlib.util.spec_from_file_location('onset_reduction', R / 'reduce.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
reduction = module.reduce()
assert reduction == read(R / 'reduction.json')
captured = {}
def capture_write(path, text, *args, **kwargs):
    assert path == R / 'independent-verification.json', path
    captured['verification'] = json.loads(text)
    return len(text)
with patch.object(Path, 'write_text', capture_write):
    runpy.run_path(str(R / 'verify.py'), run_name='__main__')
assert captured['verification'] == read(R / 'independent-verification.json')
readout = torch.load(R / 'runtime/scores/readout.pt', weights_only=True, map_location='cpu')
scaled_checks = 0
for window in read(R / 'panel.json')['windows']:
    tensors = torch.load(R / f"runtime/scores/{window['name']}.pt", weights_only=True, map_location='cpu')
    for name, t in tensors.items():
        z = t['logits'].double().clone()
        z[:, readout['coordinate_ids']] = (z[:, readout['coordinate_ids']] * readout['factors']).float().double()
        ids = torch.tensor(t['token_ids'])
        likelihood = (z[torch.arange(len(ids)), ids] - torch.logsumexp(z[:-1], dim=1)).sum().item()
        expected = reduction['windows'][window['name']]['candidates'][name]['equal_norm_teacher_forced_sum_logprob']
        assert abs(likelihood - expected) < 1e-10
        scaled_checks += 1
terminal = read(R / 'terminal.json')
assert all(x['exit'] == 0 and not Path('/proc', str(x['pid'])).exists()
           for x in terminal['owned_processes'])
for b in bindings:
    assert hashlib.sha256(Path(b['path']).read_bytes()).hexdigest() == b['sha256']
receipt = dict(status='passed', bound_files=len(bindings),
               reduction_exact=True, independent_verifier_exact=True,
               equal_norm_row_likelihood_checks=scaled_checks,
               native_parity_positions=len(reduction['native_parity']),
               max_parity_error=max(x['max_abs'] for x in reduction['native_parity']),
               max_error_margin_ratio=max(x['twice_error_over_margin'] for x in reduction['native_parity']),
               candidate_bytes_preserved=True, new_model_forwards=0,
               owned_producers_ended=True, candidate_cost=terminal['cost'])
(R / 'coordination/lead-verification.json').write_text(json.dumps(receipt, indent=2)+'\n')
print(json.dumps(receipt))
