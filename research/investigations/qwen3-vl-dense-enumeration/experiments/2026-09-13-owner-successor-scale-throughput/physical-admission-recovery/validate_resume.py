"""Lead's deterministic check of saved small-batch review proposals, not labels."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-13-owner-successor-scale-throughput')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(batch):
    directory = ROOT / f'physical-admission-recovery-resume/batch-{batch:02}'
    path = directory / 'input.json'
    manifest = json.loads((directory.parent / 'manifest-v1.json').read_text())
    assigned = next(item for item in manifest['batches'] if item['batch'] == batch)
    assert sha(path) == assigned['input']['sha256']
    packet = json.loads(path.read_text())
    completion = directory / 'completion.json'
    assert completion.is_file(), 'terminal completion required before lead verification'
    decisions_path = directory / 'decisions.jsonl'
    decisions = [json.loads(line) for line in decisions_path.read_text().splitlines() if line.strip()]
    expected = {group['visual_group_id']: group['job_ids'] for group in packet['groups']}
    assert len(decisions) == len(expected) == 10
    assert {d['visual_group_id'] for d in decisions} == set(expected)
    positive = []
    views = set()
    jobs = []
    # Only files explicitly bound to this group's sample may support its judgment.
    for decision in decisions:
        group = decision['visual_group_id']
        allowed = {r['path']: r['sha256'] for r in packet['cards'] + packet['crops']
                   if r['visual_group_id'] == group}
        representative = next(item for item in packet['groups'] if item['visual_group_id'] == group)['representative_card']['binding']
        allowed[representative['path']] = representative['sha256']
        for row in packet['rows']:
            if row['visual_group_id'] == group:
                allowed[row['image']['path']] = row['image']['sha256']
        assert decision['viewed'], 'a saved assessment must bind an actual viewed sample'
        for view in decision['viewed']:
            assert view['path'] in allowed, (group, 'unbound or other-sample evidence')
            assert view['detail'] == 'original'
            assert sha(view['path']) == view['sha256'] == allowed[view['path']]
            views.add(view['path'])
        actual = [job['job_id'] for job in decision['jobs']]
        assert actual == expected[group], (group, 'exact-job coverage/order')
        jobs.extend(actual)
        for job in decision['jobs']:
            for side, newness in [('c', 'not_seen_in_h'), ('w', 'not_seen_in_h_or_c')]:
                value = job[side]
                assert value['status'] in {'HOLD', 'candidate_accept'}
                assert value['reason'] and value['newness_reason']
                if value['status'] == 'candidate_accept':
                    assert value['newness'] == newness, (job['job_id'], side, 'unresolved newness')
            if job['c']['status'] == job['w']['status'] == 'candidate_accept':
                assert job['pair_distinct'] is True and job['pair_reason']
                positive.append(job['job_id'])
    assert len(jobs) == len(set(jobs)) == len(assigned['job_ids'])
    assert set(jobs) == set(assigned['job_ids'])
    return {'batch': batch, 'groups': len(decisions), 'jobs': len(jobs),
            'candidate_jobs': positive, 'hold_jobs': len(jobs) - len(positive),
            'bound_viewed_files': len(views), 'input_sha256': sha(path),
            'decisions_sha256': sha(decisions_path), 'completion_sha256': sha(completion),
            'status': 'lead_verified_saved_proposal_integrity_not_training_admission'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('batches', type=int, nargs='+')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = {'schema': 'physical_review_resume.lead_integrity_check.v1',
              'batches': [validate(batch) for batch in args.batches],
              'boundary': 'Checks saved evidence and criteria consistency; not independent visual verification or root admission.'}
    if args.output:
        with args.output.open('x') as stream:
            json.dump(result, stream, indent=2)
            stream.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
