"""One-arm CPU acceptance arithmetic; run from the research-probes worktree."""
import json
import math
import numpy as np
import torch
from probes.dora_owner_learning.selective_preservation import (
    OUTPUT, ROOT, selective_loss, publish, file_hash, summarize_logits,
)

p = json.loads((OUTPUT / 'inputs.json').read_text())
r = json.loads((OUTPUT / 'receipt.json').read_text())
assert r['updates'] == 23 and [x['update'] for x in r['dose']] == list(range(1, 24))
assert r['frozen_tensor_hash_before'] == r['frozen_tensor_hash_after']
assert r['resources']['actual_model_forwards'] == 96
checks, analytic, total = [], [], 0.
for j, c in enumerate(p['cases']):
    t = p['trajectories'][c['case_id']]
    a, inds, index = t['action_ids'], t['preservation_positions'], c['action_index']
    vocab = r['references'][j]['shape'][1]
    ref = torch.full((len(inds), vocab), -1000.)
    ref[:, 0], ref[:, 1] = math.log(.8), math.log(.2)
    logits = torch.full((len(a), vocab), -1000.)
    q = (.5, .5) if j == 0 else (.8, .2)
    logits[:, 0], logits[:, 1] = math.log(q[0]), math.log(q[1])
    ep = .25 if j == 0 else .5
    logits[index, :] = -1000.
    logits[index, 0], logits[index, c['target_token_id']] = math.log(1 - ep), math.log(ep)
    logits.requires_grad_(True)
    loss, ce, kl = selective_loss(logits, torch.tensor(a), c['entrance'], a, t['suffix_start'], inds, ref)
    loss.backward()
    expected = (torch.tensor(q) - torch.tensor([.8, .2])) * (5 / len(inds))
    assert torch.allclose(logits.grad[inds[0], :2], expected, atol=1e-7)
    total += float(loss.detach())
    analytic.append(dict(case_id=c['case_id'], preserved_states=len(inds), half_loss=float(loss.detach()),
        first_anchor_gradient=logits.grad[inds[0], :2].tolist(), expected_gradient=expected.tolist()))
    del ref, logits, loss, ce, kl
    initial = np.load(OUTPUT / 'scores' / f"step-00-{c['image_id']}.npy", allow_pickle=False)
    prior = np.load(ROOT / '2026-09-10-native-entrance-ce-feasibility/training/scores' / f"step-00-{c['image_id']}.npy", allow_pickle=False)
    final = np.load(OUTPUT / 'scores' / f"step-23-{c['image_id']}.npy", allow_pickle=False)
    s = summarize_logits(final, c['entrance'])
    s['target_id'] = c['target_token_id']
    assert s == r['final_scores'][c['case_id']]
    assert np.array_equal(initial, prior)
    checks.append(dict(case_id=c['case_id'], source_initial_exact_equal=True,
        final_score_replay=True, excluded_positions=t['excluded_positions'],
        EOS_position=len(a) - 1, EOS_preserved=len(a) - 1 in inds))
assert abs(total - 2.00344455595) < 2e-6
report = dict(status='candidate_for_lead_acceptance', receipt_sha256=file_hash(OUTPUT / 'receipt.json'),
    tests=dict(command='OMP_NUM_THREADS=4 python -m pytest -q probes/dora_owner_learning/tests/test_selective_preservation.py', passed=6, exit_code=0),
    analytic_fixture=dict(total=total, expected=2.00344455595, cases=analytic), identity_and_mask_checks=checks,
    frozen_bytes_unchanged=True, exact_fixed23_updates=True, process_exit_code=0,
    source_parameter_delta_l2=r['dose'][-1]['source_parameter_delta_l2'], last_recorded_KL_is_pre_update23=True,
    acceptance_commands=['python -m pytest -q probes/dora_owner_learning/tests/test_selective_preservation.py',
                         'python -m probes.dora_owner_learning.selective_preservation verify'])
path = OUTPUT / 'worker_verification.json'
if path.exists():
    assert json.loads(path.read_text()) == report
else:
    publish(path, report)
print(json.dumps(report))
