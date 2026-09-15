"""Partition the unchanged incomplete PAR review into disjoint ten-group tasks."""

import hashlib
import json
from pathlib import Path


ROOT = Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-13-owner-successor-scale-throughput')


def binding(path):
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write('\n')


def main():
    source = ROOT / 'physical-admission-recovery/review-index-v1.json'
    assert binding(source)['sha256'] == '59c402bc073a8fb3c2494be02a645bb178eb427bd410597c5373ac3a7d1f4cc8'
    index = json.loads(source.read_text())
    cards_path = ROOT / 'physical-admission-recovery/card-manifest.json'
    crops_path = ROOT / 'physical-admission-recovery/individual-crop-manifest-v1.json'
    cards = json.loads(cards_path.read_text())
    crops = json.loads(crops_path.read_text())['records']
    groups = index['groups']
    assert len(groups) == 80 and len(index['rows']) == 98
    manifest = {'schema': 'physical_review_recovery_batches.v1',
                'source_index': binding(source), 'card_manifest': binding(cards_path),
                'crop_manifest': binding(crops_path), 'batches': [],
                'policy': 'No inferred Luna labels; existing images reused. Each decision saved before viewing the next sample.'}
    covered = []
    for start in range(0, 80, 10):
        selected = groups[start:start + 10]
        ids = {group['visual_group_id'] for group in selected}
        rows = [row for row in index['rows'] if row['visual_group_id'] in ids]
        packet = {'schema': 'physical_review_recovery_batch.v1',
                  'status': 'pending_individual_assessment',
                  'source_index': binding(source),
                  'groups': selected, 'rows': rows,
                  'cards': [card for card in cards if card['visual_group_id'] in ids],
                  'crops': [crop for crop in crops if crop['visual_group_id'] in ids]}
        path = ROOT / f'physical-admission-recovery-resume/batch-{start // 10 + 1:02}/input.json'
        write_new(path, packet)
        manifest['batches'].append({'batch': start // 10 + 1, 'input': binding(path),
                                    'groups': sorted(ids), 'job_ids': [row['job_id'] for row in rows]})
        covered.extend(row['job_id'] for row in rows)
    assert len(covered) == len(set(covered)) == 98
    assert set(covered) == {row['job_id'] for row in index['rows']}
    write_new(ROOT / 'physical-admission-recovery-resume/manifest-v1.json', manifest)
    print(json.dumps({'batches': 8, 'groups': 80, 'jobs': len(covered), 'duplicate_jobs': 0}))


if __name__ == '__main__':
    main()
