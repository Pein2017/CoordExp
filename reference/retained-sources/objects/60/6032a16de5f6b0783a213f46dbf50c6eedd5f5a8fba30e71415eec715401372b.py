"""Lead-owned bounded review integration; run only after review acceptance."""
import hashlib
import json
from collections import Counter
from pathlib import Path

root = Path(__file__).resolve().parent
entries, ledger, bindings = [], [], []
reviewed = 0
for index, expected in enumerate([32, 33, 32, 31]):
    batch_path = root / f'batch-{index}.json'
    decision_path = root / f'decisions-{index}.jsonl'
    cases = json.loads(batch_path.read_text())
    decisions = [json.loads(line) for line in decision_path.read_text().splitlines()]
    assert len(cases) == expected and len(decisions) <= expected, (index, len(decisions), expected)
    by_id = {r['candidate_id']: r for r in decisions}
    assert len(by_id) == len(decisions) and set(by_id) <= {c['candidate_id'] for c in cases}
    reviewed += len(decisions)
    for case in [c for c in cases if c['candidate_id'] not in by_id]:
        decisions.append({'candidate_id': case['candidate_id'], 'image_id': case['image_id'],
                          'disposition': 'HOLD', 'proposed_action': 'none',
                          'review_status': 'not_reviewed_user_stopped_visual_expansion',
                          'reason': 'User stopped further view_images and requested convergence; no visual admission evidence.',
                          'evidence_paths': [str(batch_path)]})
    by_id = {r['candidate_id']: r for r in decisions}
    decisions = [by_id[c['candidate_id']] for c in cases]
    for path in [batch_path, decision_path]:
        bindings.append({'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    for case, decision in zip(cases, decisions):
        cid = case['candidate_id']
        assert str(case['image_id']) == str(decision['image_id'])
        assert all(Path(path).is_file() for path in decision['evidence_paths'])
        add = decision['proposed_action'] == 'add_owner'
        if add:
            assert decision['disposition'] == 'candidate'
            for key, value in {'entity': 'verified_real', 'category_status': 'verified_correct',
                               'relation': 'unlabeled_real', 'geometry': 'acceptable',
                               'uniqueness': 'distinct_owner', 'confidence': 'high'}.items():
                assert decision[key] == value, (cid, key)
            obj = decision['accepted_object']
            bins = [int(str(x).removeprefix('<|coord_').removesuffix('|>')) for x in obj['bbox_2d']]
            assert bins == case['coord_bins'] and obj['desc'] == case['category'], cid
            assert all(0 <= x <= 999 for x in bins)
            assert bins[0] < bins[2] and bins[1] < bins[3]
        else:
            assert decision['disposition'] in ['HOLD', 'rejected']
            assert decision['proposed_action'] == 'none'
        entry = {
            'admission_id': f'review-{cid}', 'image_id': int(case['image_id']),
            'candidate_id': cid, 'disposition': 'lead-accepted' if add else decision['disposition'],
            'action': 'add_owner' if add else 'none',
            'owner_id': f'source256-unlabeled-{cid}' if add else None,
            'provenance': {'evidence': decision['evidence_paths'], 'reason': decision['reason'],
                           'review_decisions': str(decision_path),
                           'lead_acceptance': 'Exact identity/box/evidence validation; visual reviewer decisions and bounded lead visual spot-checks; no detector auto-admission.'},
        }
        if add:
            entry['object'] = {'desc': obj['desc'], 'bbox_2d': [f'<|coord_{x}|>' for x in bins]}
        entries.append(entry)
        ledger.append({**decision, 'lead_disposition': entry['disposition']})
assert len({r['candidate_id'] for r in entries}) == len(entries) == 128
assert max(Counter(r['image_id'] for r in entries).values()) <= 2
exclusions_path = root / 'seed-exclusions.json'
exclusions = json.loads(exclusions_path.read_text())['entries']
assert len(exclusions) == 1 and exclusions[0]['owner_id'] == '1724857'
entries.extend(exclusions)
summary = {'selected': 128, 'reviewed': reviewed, 'selected_unreviewed_hold': 128-reviewed,
           'images': len({r['image_id'] for r in entries if r['candidate_id']}),
           'actions': dict(Counter(r['action'] for r in entries)),
           'dispositions': dict(Counter(r['disposition'] for r in entries)),
           'bounded_lead_visual_spotchecks': ['candidate-228d38e8d5e77be7b03b', 'candidate-b0e195f224beeddf932f',
               'candidate-21c6872fdc47ec25e319', 'candidate-fcc0a89c9687e706e597', 'candidate-81cdfd12c3916f72042d'],
           'lead_direct_review': 'decisions-3.jsonl (31 cases)', 'bindings': bindings,
           'unreviewed_hypotheses': 3513-reviewed, 'unreviewed_disposition': 'HOLD',
           'visual_stop_authority': 'User: 可以停下view_images了; 结论应该可以收敛了'}
admissions = {'schema': 'source256.review_admissions.v1', 'version': 'visual-admitted-v1',
              'authority': 'Lead accepted under frozen Source256 128-hypothesis visual policy; original annotations preserved.',
              'entries': entries}
for name, value in [('review-admissions.json', admissions), ('acceptance.json', summary)]:
    path = root / name
    assert not path.exists(), path
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
ledger_path = root / 'lead-reviewed-ledger.jsonl'
assert not ledger_path.exists()
ledger_path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in ledger))
print(json.dumps({k: v for k, v in summary.items() if k not in ['bindings', 'bounded_lead_visual_spotchecks']}))
