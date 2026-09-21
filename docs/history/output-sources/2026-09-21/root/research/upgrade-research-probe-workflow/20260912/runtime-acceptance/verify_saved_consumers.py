"""Cold CPU-only check of the real-run saved arrays through existing scorers."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--check-only', action='store_true', help='Verify and print without creating a receipt')
    args = parser.parse_args()
    sys.path.insert(0, str(args.source.resolve()))
    import numpy as np
    import torch
    from probes.dora_owner_learning.branch_bridge import summarize_logits
    from probes.dora_owner_learning.route_access import score_logits

    torch.set_num_threads(4)
    run = args.run
    runtime = json.loads((run / 'receipt.json').read_text())
    assert runtime['status'] == 'completed' and runtime['process_exit_code'] == 0
    old = json.loads((args.baseline / 'old/source256.json').read_text())['rows'][0]
    paired = json.loads((run / 'paired-diagnostics.json').read_text())
    zero = json.loads((run / 'zero-score.json').read_text())
    targets = torch.tensor(old['action_token_ids'], dtype=torch.long)
    entry = paired['entry']
    assert entry == runtime['update']['entry']
    assert entry['state_ids'] == targets[:entry['action_index']].tolist()
    assert entry['A_id'] == int(targets[entry['action_index']])
    arrays = {name: np.load(run / file, allow_pickle=False) for name, file in (
        ('old', 'old-zero-logits.npy'), ('new', 'new-zero-logits.npy'), ('after', 'after-update-logits.npy'))}
    assert np.array_equal(arrays['old'], arrays['new'])
    assert all(a.dtype == np.float32 and a.ndim == 2 and a.shape[0] == len(targets)
               and np.isfinite(a).all() for a in arrays.values())
    scores = {name: score_logits(torch.from_numpy(values), targets,
              prompt_length=old['generation_prefix_length']) for name, values in arrays.items()}
    assert zero['old'] == zero['new'] == paired['token_scores_before']
    assert scores['old'] == scores['new']
    rounding = {}
    for phase, observed, saved in (
        ('before', scores['new'], paired['token_scores_before']),
        ('after', scores['after'], paired['token_scores_after']),
    ):
        # Same-device old/current scores remain exact. Cross-device log-softmax
        # reductions use installed torch.testing's default FP32 tolerance.
        for field in observed:
            if field not in ('positions', 'sum_logprob', 'mean_logprob'):
                assert observed[field] == saved[field], (phase, field)
        for current, recorded in zip(observed['positions'], saved['positions'], strict=True):
            assert {k: v for k, v in current.items() if k != 'logprob'} == {
                k: v for k, v in recorded.items() if k != 'logprob'}
        current_logp = torch.tensor([p['logprob'] for p in observed['positions']], dtype=torch.float32)
        saved_logp = torch.tensor([p['logprob'] for p in saved['positions']], dtype=torch.float32)
        torch.testing.assert_close(current_logp, saved_logp)
        assert observed['sum_logprob'] == sum(p['logprob'] for p in observed['positions'])
        assert saved['sum_logprob'] == sum(p['logprob'] for p in saved['positions'])
        assert observed['mean_logprob'] == observed['sum_logprob'] / len(targets)
        assert saved['mean_logprob'] == saved['sum_logprob'] / len(targets)
        rounding[phase] = dict(max_token_logprob_abs_delta=float((current_logp - saved_logp).abs().max()),
            sum_logprob_abs_delta=abs(observed['sum_logprob'] - saved['sum_logprob']),
            mean_logprob_abs_delta=abs(observed['mean_logprob'] - saved['mean_logprob']),
            non_logprob_fields_exact=True)
    before = summarize_logits(arrays['new'][entry['action_index']], entry)
    after = summarize_logits(arrays['after'][entry['action_index']], entry)
    assert before == paired['before'] and after == paired['after']
    bound = json.loads((run / 'executed-source.json').read_text())['files']
    recorded_root = Path(runtime['source_root']).resolve()
    verified_root = args.source.resolve()
    for row in bound:
        relative = Path(row['path']).relative_to(recorded_root)
        assert sha(verified_root / relative) == row['sha256'] == sha(row['staged'])
    files = ['old-zero-logits.npy', 'new-zero-logits.npy', 'after-update-logits.npy',
             'paired-diagnostics.json', 'zero-score.json', 'receipt.json', 'executed-source.json']
    receipt = dict(status='completed', scope='Cold saved FP32 arrays to unchanged conditional scorers; no model loading or generation.',
        script_sha256=sha(__file__), files={name: sha(run / name) for name in files},
        baseline_sha256=sha(args.baseline / 'old/source256.json'),
        array_shape=list(arrays['old'].shape), exact_zero_logits=True,
        exact_same_backend_old_current_token_scores=True, exact_saved_branch_scores=True,
        cross_device_token_score_policy='Installed torch.testing.assert_close default float32 tolerance; all non-logprob fields exact; sum/mean recomputed from per-token values.',
        cpu_gpu_rounding=rounding,
        recorded_source_root=str(recorded_root), verified_source_root=str(verified_root),
        executed_source_hashes_verified=len(bound), model_loads=0, model_forwards=0,
        observed_target_logprob_before=before['logprob'], observed_target_logprob_after=after['logprob'])
    if not args.check_only:
        with (run / 'cold-consumer-receipt.json').open('x') as stream:
            json.dump(receipt, stream, indent=2, allow_nan=False)
            stream.write('\n')
    print(json.dumps(receipt, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
