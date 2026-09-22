"""Read-only projection of saved endpoints for the 2026-09-14 consultation."""
import collections
import hashlib
import json
from pathlib import Path

BASE = Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-13-owner-successor-scale-throughput/training')


def read(path):
    return json.loads(path.read_text())


def binding(path):
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


packet_path = BASE / 'inputs-sealed-v1.json'
packet = read(packet_path)
old_path = Path(packet['n16_training_input']['path'])
old_records = read(old_path)['positive_records']
new_records = [p['c_record'] for p in packet['new_packages']]
records = old_records + new_records
groups = collections.defaultdict(list)
for record in records:
    key = (record['example_id'], tuple(record['prompt_token_ids']), tuple(record['prefix_token_ids']))
    groups[key].append(record)

out = {
    'status': 'retained_literal_score_projection_not_physical_owner_review',
    'inputs': [binding(packet_path), binding(old_path)],
    'records': len(records),
    'unique_image_prompt_prefix_conditions': len(groups),
    'multiple_distinct_literal_target_conditions': sum(
        len({tuple(r['target_token_ids']) for r in rows}) > 1 for rows in groups.values()
    ),
    'arms': {},
    'limits': [
        'Literal target argmax is not semantic owner recovery or root reachability.',
        'Multiple targets at one history are alternatives, not proof of erroneous labels.',
        'Distinct literal rows are not an independently re-reviewed physical identity join.',
        'A/B terminal comparison is not a within-arm learning-dose trajectory.',
    ],
}
for arm in ('A', 'B'):
    path = BASE / f'full-{arm}-v2/receipt.json'
    receipt = read(path)
    scores = receipt['final_live_positive_scores']
    assert set(scores) == {r['record_id'] for r in records}
    summaries = {}
    for name, bank in [('old', old_records), ('new', new_records), ('all', records)]:
        ids = {r['record_id'] for r in bank}
        selected = [rows for rows in groups.values() if any(r['record_id'] in ids for r in rows)]
        vals = [scores[rid] for rid in ids]
        hit = lambda rid: scores[rid]['argmax_target_tokens'] == scores[rid]['token_count']
        summaries[name] = {
            'records': len(ids),
            'conditions': len(selected),
            'literal_full_argmax_rows': sum(hit(rid) for rid in ids),
            'conditions_with_any_literal_full_argmax': sum(
                any(hit(r['record_id']) for r in rows if r['record_id'] in ids) for rows in selected
            ),
            'target_tokens': sum(v['token_count'] for v in vals),
            'argmax_target_tokens': sum(v['argmax_target_tokens'] for v in vals),
            'summed_nll': -sum(v['sum_logprob'] for v in vals),
        }
    out['arms'][arm] = {'receipt': binding(path), 'updates': receipt['updates'], 'banks': summaries}
out['saved_tensor_files_in_training_tree'] = [
    {'path': str(p.relative_to(BASE)), 'bytes': p.stat().st_size}
    for p in sorted(BASE.rglob('*')) if p.is_file() and p.suffix in {'.safetensors', '.pt', '.pth'}
]
print(json.dumps(out, ensure_ascii=False, indent=2))
