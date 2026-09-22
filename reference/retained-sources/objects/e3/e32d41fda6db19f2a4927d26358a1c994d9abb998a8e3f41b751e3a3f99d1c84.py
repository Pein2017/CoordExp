"""Task-local CPU verification and closeout readout; no model construction."""
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from probes.dora_owner_learning.reward_rows import _gt_objects, _pred_objects
from src.data.geometry import iou_xyxy

R = Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-10-native-entrance-ce-feasibility')
BRIDGE = R.parent / '2026-09-10-verified-branch-update-bridge'
def read(path):
    return json.loads(path.read_text())
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

t = read(R / 'training/receipt.json')
m = read(R / 'evaluation/manifest.json')
records = read(R / 'evaluation/execution/consumer.json')
reduction = read(R / 'evaluation/execution/reduction.json')
diag = read(R / 'evaluation/repeat-diagnostics.json')
assert t['updates'] == 23 and all(x['stop_reason'] is None for x in t['dose'][:-1])
assert all(v['A_vs_best_other_margin'] >= .1 for v in t['final_scores'].values())
assert t['frozen_tensor_hash_before'] == t['frozen_tensor_hash_after']
assert t['source_adapter']['root'] != t['adapter']['root']
first_top1 = next(x['update'] for x in t['dose'] if all(v['rank_min'] == 1 and v['A_vs_best_other_margin'] > 0 for v in x['final_scores'].values()))
for identity in (t['source_adapter'], t['source_embedding'], t['adapter']):
    for f in identity['files']:
        assert sha(Path(identity['root']) / f['relative_path']) == f['sha256']
for case in t['cases']:
    stem = case['case_id'].replace(':', '_')
    initial = np.load(R / f"training/scores/step-00-{case['image_id']}.npy", allow_pickle=False)
    previous = np.load(BRIDGE / f'execution/source-{stem}-entry-logits.npy', allow_pickle=False)
    final = np.load(R / f"training/scores/step-23-{case['image_id']}.npy", allow_pickle=False)
    cold = np.load(R / f'evaluation/execution/{stem}-cold-entry-logits.npy', allow_pickle=False)
    assert np.array_equal(initial, previous) and np.array_equal(final, cold)
    target = case['target_token_id']
    assert final.argmax() == target
    other = final.copy(); other[target] = -np.inf
    assert float(final[target] - other.max()) >= .1

assert len(records) == 18 and all(x['prefix_ids'] == [] and x['forced_ids'] == [] for x in records)
for split in ('train', 'guard'):
    xs = [x for x in records if x['split'] == split]
    ys = [x for x in m['records'] if x['split'] == split]
    s = reduction['splits'][split]
    for threshold in ('50', '60', '80'):
        tp = sum(x['score'][threshold]['tp'] for x in xs)
        fp = sum(x['score'][threshold]['fp'] for x in xs)
        fn = sum(x['score'][threshold]['fn'] for x in xs)
        assert s['candidate'][threshold] == dict(tp=tp, fp=fp, fn=fn, f1=2*tp/(2*tp+fp+fn))
        before = sum(x['baseline_score'][threshold]['tp'] for x in ys)
        changes = s['owner_changes'][threshold]
        assert tp - before == changes['gained'] - changes['lost']

boat = next(x for x in records if x['example_id'].endswith('007116'))
preds, invalid = _pred_objects(boat['parsed']); assert invalid == 0
gts = _gt_objects(boat['parsed'], row_id=boat['example_id'])
gt_ids = [str(x['object_id']) for x in boat['parsed']['gt']]
repeat_ids = [i for i, (_, p) in enumerate(preds) if any(iou_xyxy(p, q) > .95 for _, q in preds[:i])]
assert len(repeat_ids) == 21 == diag['strict_later_repeats']
assert sum(x['strict_later_repeat_count'] for x in diag['clusters']) == 21
clusters = []
for cluster in diag['clusters']:
    members = cluster['pred_indices']
    owners = sorted({oid for oid, (category, box) in zip(gt_ids, gts) for i in members
                     if category == preds[i][0] and iou_xyxy(box, preds[i][1]) >= .5})
    clusters.append(dict(category_counts=cluster['category_counts'], boxes=len(members),
        strict_repeats=cluster['strict_later_repeat_count'], direct_GT_owners=owners))

def coord_bins(record, pred_index):
    pred = record['parsed']['pred'][pred_index]
    span = record['text'][pred['char_start']:pred['char_end']]
    coords = [int(x) for x in re.findall(r'<\|coord_(\d+)\|>', span)]
    assert len(coords) == 4
    return coords

assert sum(coord_bins(boat, i)[0] == 291 for i in range(len(preds))) == 27
targets = []
for row in records:
    if row['split'] != 'train':
        continue
    owner = row['training_readout']['owner']
    match = next(x for x in row['score']['50']['matches'] if x['owner'] == owner)
    targets.append(dict(example_id=row['example_id'], owner=owner, pred_index=match['pred_index'],
        coords=coord_bins(row, match['pred_index']), iou=match['iou'],
        target_thresholds=row['training_readout']['target_global'],
        exact_entrance_reached=row['training_readout']['exact_entrance_reached'],
        first_divergence=row['training_readout']['first_divergence_from_Source']))

eval_term = read(R / 'evaluation/execution/terminal.json')
assert eval_term['status'] == 'completed' and eval_term['continuations'] == 18
model_seconds = t['resources']['cumulative_model_seconds'] + eval_term['elapsed_seconds']
result = dict(schema='native_entrance_ce.lead_readout.v1', status='verified',
    selected_update=23, first_joint_actual_top1_update=first_top1,
    initial_Source_logits_exact=True, final_saved_cold_logits_exact=True,
    source_embedding_and_candidate_bytes_verified=True, frozen_parameter_hash_unchanged=True,
    natural_outputs=18, pooled_arithmetic_verified=True, repeat_clusters=clusters,
    native_target_rows=targets, valid_boat_rows_with_x1_291=27,
    model_seconds=model_seconds, model_gpu_hours=model_seconds/3600,
    scientific_disposition='fixed_state_and_natural_target_learning_with_collateral_burden_and_guard_regression',
    checkpoint_promotion='not_promoted', verification_script_sha256=sha(Path(__file__)))
path = R / 'lead-readout.json'
if path.exists():
    assert read(path) == result
else:
    with path.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True); stream.write('\n')
print(json.dumps(result, indent=2))
