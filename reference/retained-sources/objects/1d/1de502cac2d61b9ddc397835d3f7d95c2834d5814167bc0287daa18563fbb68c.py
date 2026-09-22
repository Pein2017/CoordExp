"""CPU-only root acceptance of the frozen candidate; preserve candidate bytes."""
import contextlib
import hashlib
import io
import json
import runpy
import sys
from pathlib import Path
from unittest.mock import patch

import torch

R = Path(__file__).resolve().parent.parent
sys.path.insert(0, '/data/CoordExp/.worktrees/research-probes')
torch.set_num_threads(4)
read = lambda p: json.loads(p.read_text())
manifest = read(R / 'manifest.json')
bindings = [b for k in ('files', 'external_sources', 'frozen_inputs') for b in manifest[k]]
def check_bindings():
    for b in bindings:
        p = Path(b['path'])
        assert p.stat().st_size == b['size_bytes'], p
        assert hashlib.sha256(p.read_bytes()).hexdigest() == b['sha256'], p
check_bindings()
captured = {}
def capture_write(path, value, *args, **kwargs):
    assert path in (R / 'reduction.json', R / 'physical-score.json'), path
    captured[path.name] = json.loads(value)
    return len(value)
with patch.object(Path, 'write_text', capture_write), contextlib.redirect_stdout(io.StringIO()):
    runpy.run_path(str(R / 'reduce.py'), run_name='__main__')
    runpy.run_path(str(R / 'physical_score.py'), run_name='__main__')
for name, value in captured.items():
    assert value == read(R / name), name
assert len(captured) == 2
result = read(R / 'result.json')
rows = captured['reduction.json']['paths']
assert rows == result['paths']
assert len(rows) == 14 and len({tuple(p['row_token_ids']) for p in rows}) == 14
assert all(p['stop'] == 'complete' for p in rows)
native = next(p for p in rows if p['id'] == 'b0-p0')
assert all(t['rank'] == 1 and t['chosen_minus_best_other'] > 0 for t in native['token_scores'])
for branch in result['branches']:
    ps = [p for p in rows if p['branch'] == branch['branch']]
    best = max(ps, key=lambda p: p['full_row_logprob'])
    assert best['id'] == branch['best'] and best['box'] == branch['best_box']
    assert abs(best['full_row_logprob'] - branch['best_logp']) < 1e-12
    assert abs(best['full_row_logprob'] - native['full_row_logprob'] - branch['best_minus_original_greedy']) < 1e-12
    if branch['branch'] != 0:
        assert all(p['full_row_logprob'] < native['full_row_logprob'] for p in ps)
annotation = read(R / 'annotation-identity.json')
assert annotation['count'] == 63 and all(annotation['checks'].values())
terminal = read(R / 'terminal.json')
assert terminal['failed_model_cells'] == 0
for job in terminal['jobs']:
    assert job['exit'] == 0 and not Path('/proc', str(job['pid'])).exists()
    rec = read(R / 'runtime' / str(job['branch']) / 'receipt.json')
    assert rec['fresh_parent_requery_exact'] and rec['parameters_unchanged']
check_bindings()
receipt = dict(status='passed', binding_entries=len(bindings),
               unique_bound_paths=len({b['path'] for b in bindings}),
               reduction_exact=True, physical_score_replay_exact=True,
               unique_complete_paths=len(rows), ranking_verified=True,
               native_greedy_all_competitor_margins_positive=True,
               native_min_margin=min(t['chosen_minus_best_other'] for t in native['token_scores']),
               max_complete_path_rescore_error=max(p['rescore_error'] for p in rows),
               annotation_positive_count=63, candidate_bytes_preserved=True,
               owned_producers_ended=True, new_model_forwards=0, cost=result['cost'])
(R / 'coordination/lead-verification.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt))
