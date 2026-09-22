"""Serialize root's decision on an exact, reviewed proposal snapshot; no new selection policy."""

import hashlib
import json
from pathlib import Path

ROOT = Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-13-owner-successor-scale-throughput')
SUMMARY_SHA = '63fdd55b8b1b958045b39ccc75d03a26f17959092627d7012c7a2a98dd55fa8a'
AUDIT_SHA = {
    1: 'd83350b77bda0765372bdbe866d023ba5145b1fcf342fb39b6609a3ad477d673',
    2: 'd339c8501187f08a85a7f51dd1d04fa7537e7564013670bbae82051f26fad5f1',
}
# Root read the bound c/w rationales, literal classes, exact-history identities,
# and the sole targeted audit. It admits this exact 60-group snapshot except
# the demonstrated same-person sub-box. No later proposal can enter implicitly.
ROOT_HOLD = {'PAM-0105': 'Literal person w is a sub-box of c same person, not a new handbag/person; targeted visual audit confirms no distinct successor. Unknown-neutral HOLD; no negative target.'}


def bind(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    summary_path = ROOT / 'physical-admission-join/final-v1/candidate-summary-v1.json'
    assert bind(summary_path)['sha256'] == SUMMARY_SHA
    summary = json.loads(summary_path.read_text())
    assert summary['counts']['completed_proposal_jobs'] == 502
    assert summary['counts']['pending_recovery_jobs'] == 0
    assert not summary['pending_recovery']['invalid_decisions']
    index_path = Path(summary['consumer_review_index']['path'])
    assert bind(index_path)['sha256'] == summary['consumer_review_index']['sha256']
    index = json.loads(index_path.read_text())
    assert index['status'] == 'review_complete'
    rows = {row['job_id']: row for row in index['rows']}
    proposal_groups = {group['visual_group_id']: group for group in summary['candidate_positive_groups']}
    assert len(proposal_groups) == 60 and set(ROOT_HOLD) <= set(proposal_groups)
    manifest_path = ROOT / 'physical-admission-targeted-audit/manifest-v1.json'
    manifest = json.loads(manifest_path.read_text())
    audit_positive_jobs = {}
    audits = []
    for batch in manifest['batches']:
        input_path = Path(batch['input']['path'])
        assert bind(input_path) == batch['input']
        packet = json.loads(input_path.read_text())
        decision_path = input_path.parent / 'decisions.jsonl'
        assert bind(decision_path)['sha256'] == AUDIT_SHA[batch['batch']]
        assert (input_path.parent / 'completion.json').is_file()
        decisions = [json.loads(line) for line in decision_path.read_text().splitlines() if line.strip()]
        assert len(decisions) == len(batch['groups'])
        assert {d['visual_group_id'] for d in decisions} == set(batch['groups'])
        for decision in decisions:
            group = decision['visual_group_id']
            expected = next(g['job_ids'] for g in packet['groups'] if g['visual_group_id'] == group)
            assert [j['job_id'] for j in decision['jobs']] == expected
            allowed = {r['path']: r['sha256'] for r in packet['cards'] + packet['crops'] if r['visual_group_id'] == group}
            for row in packet['rows']:
                if row['visual_group_id'] == group:
                    allowed[row['image']['path']] = row['image']['sha256']
            assert decision['viewed']
            for view in decision['viewed']:
                assert view['detail'] == 'original' and view['path'] in allowed
                assert bind(view['path'])['sha256'] == view['sha256'] == allowed[view['path']]
            positives = []
            for job in decision['jobs']:
                for side, newness in [('c', 'not_seen_in_h'), ('w', 'not_seen_in_h_or_c')]:
                    assert job[side]['status'] in {'candidate_accept', 'HOLD'}
                    assert job[side]['reason'] and job[side]['newness_reason']
                    if job[side]['status'] == 'candidate_accept':
                        assert job[side]['newness'] == newness
                if job['c']['status'] == job['w']['status'] == 'candidate_accept':
                    assert job['pair_distinct'] is True and job['pair_reason']
                    positives.append(job['job_id'])
            audit_positive_jobs[group] = positives
        audits.append({'input': bind(input_path), 'decisions': bind(decision_path),
                       'completion': bind(input_path.parent / 'completion.json')})
    assert len(audit_positive_jobs) == 13
    assert {group for group, jobs in audit_positive_jobs.items() if not jobs} == set(ROOT_HOLD)
    decisions = []
    for group in index['groups']:
        group_id = group['visual_group_id']
        decision = {'visual_group_id': group_id, 'disposition': 'hold',
                    'reason': ROOT_HOLD.get(group_id, 'Unresolved physical c/w trust under the frozen review; neutral, not a negative target.')}
        if group_id in proposal_groups and group_id not in ROOT_HOLD:
            candidates = proposal_groups[group_id]['exact_candidate_job_ids']
            if group_id in audit_positive_jobs:
                candidates = [job for job in candidates if job in audit_positive_jobs[group_id]]
            assert candidates
            canonical = min(candidates, key=lambda job: (
                rows[job]['source_identity']['history_index'], rows[job]['source_identity']['candidate_index'], job))
            decision.update({
                'disposition': 'admit', 'canonical_job_id': canonical,
                'reason': 'Root accepts bound individual-image c/w physical/class/visible-extent and exact-history evidence; earliest trusted exact history/candidate chosen within the frozen visual group.',
                'c_trust': {'status': 'physically_trusted', 'supported_not_yet_covered': True},
                'w_trust': {'status': 'physically_trusted', 'immediate_successor': True},
                'alias_history': {'status': 'canonical_exact_job_history', 'canonical_job_id': canonical},
                'targeted_audit_used': group_id in audit_positive_jobs,
            })
        decisions.append(decision)
    assert len(decisions) == 429 and sum(d['disposition'] == 'admit' for d in decisions) == 59
    payload = {
        'schema': 'owner_successor_scale.physical_training_decisions.v1',
        'status': 'root_decisions_complete',
        'pool': bind(ROOT / 'supply/pool.json'),
        'confirmation_selection': bind(ROOT / 'evaluation/confirmation-selection.json'),
        'review_indexes': [bind(index_path)], 'decisions': decisions,
        'evidence': {'proposal_summary': bind(summary_path),
                     'recovery_integrity': bind(ROOT / 'physical-admission-recovery-resume/lead-integrity-all-v1.json'),
                     'targeted_audit_manifest': bind(manifest_path), 'targeted_audits': audits},
        'authority': 'Root research admission of this exact AI-reviewed snapshot, not human exhaustive annotation, new owner count, checkpoint promotion or publication.',
        'selection_rule': 'Existing consumer: frozen pool order, distinct-image-first, max two packages per image, 64 image /128 package caps, 32 package /16 image floor.',
        'negative_labels_created': 0,
    }
    output = ROOT / 'physical-admission/root-training-decisions-v1.json'
    with output.open('x') as stream:
        json.dump(payload, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'root_decisions': bind(output), 'admitted_groups_before_selection': 59,
                      'held_groups': 370, 'targeted_audit_rejected_pair': 'PAM-0105'}))


if __name__ == '__main__':
    main()
