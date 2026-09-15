"""Freeze one bounded recheck for demonstrated row/rationale mismatch and extent ambiguity."""

import copy
import json
from pathlib import Path

from resume_batches import ROOT, binding, write_new


def main():
    source = ROOT / 'physical-admission-join/final-v1/consumer-review-index-v1.json'
    index = json.loads(source.read_text())
    old = json.loads((ROOT / 'physical-admission-remainder/batch-03/physical-review-v1.json').read_text())
    old_rows = {row['job_id']: row for row in old['rows']}
    recovery = json.loads((ROOT / 'physical-admission-recovery-resume/batch-08/input.json').read_text())
    batches = [
        ['PAM-0090', 'PAM-0099', 'PAM-0100', 'PAM-0101', 'PAM-0105', 'PAM-0106'],
        ['PAM-0114', 'PAM-0116', 'PAM-0118', 'PAM-0119', 'PAM-0120', 'PAR-0074', 'PAR-0075'],
    ]
    manifest = {'schema': 'physical_review.targeted_single_pass.v1', 'source_index': binding(source),
                'reason': 'PAM-0105 literal person w has a handbag rationale; same old batch positive proposals require literal-row alignment recheck. PAR-0074/75 section-only boat rationale requires full visible extent check.',
                'authority': 'Root-directed single bounded falsification pass; no new population, model calls, negative labels or criteria change.',
                'batches': []}
    for number, ids in enumerate(batches, 1):
        rows = []
        cards, crops = [], []
        for row in index['rows']:
            if row['visual_group_id'] not in ids:
                continue
            clean = copy.deepcopy({key: row[key] for key in [
                'job_id', 'visual_group_id', 'image_id', 'image', 'c', 'w', 'exact_history', 'source_identity',
            ]})
            for side in ['c', 'w']:
                clean[side].pop('physical_review', None)
            rows.append(clean)
            if row['job_id'] in old_rows:
                evidence = old_rows[row['job_id']]['review_evidence']
                cards.append({'visual_group_id': row['visual_group_id'], 'job_id': row['job_id'],
                              **evidence['individual_card_binding']})
                crops.extend({'visual_group_id': row['visual_group_id'], 'job_id': row['job_id'], **crop}
                             for crop in evidence.get('individual_crop_bindings', []))
        cards.extend(card for card in recovery['cards'] if card['visual_group_id'] in ids)
        crops.extend(crop for crop in recovery['crops'] if crop['visual_group_id'] in ids)
        groups = [{'visual_group_id': group, 'job_ids': [row['job_id'] for row in rows if row['visual_group_id'] == group]}
                  for group in ids]
        assert all(group['job_ids'] for group in groups)
        packet = {'schema': 'physical_review.targeted_single_pass_input.v1', 'source_index': binding(source),
                  'groups': groups, 'rows': rows, 'cards': cards, 'crops': crops,
                  'boundary': 'Prior judgments intentionally omitted. Verify exact literal c/w row and own history, not an adjacent visual group.'}
        path = ROOT / f'physical-admission-targeted-audit/batch-{number:02}/input.json'
        write_new(path, packet)
        manifest['batches'].append({'batch': number, 'input': binding(path), 'groups': ids,
                                    'job_ids': [row['job_id'] for row in rows]})
    write_new(ROOT / 'physical-admission-targeted-audit/manifest-v1.json', manifest)
    print(json.dumps({'batches': len(batches), 'groups': sum(map(len, batches)),
                      'job_counts': [len(batch['job_ids']) for batch in manifest['batches']]}))


if __name__ == '__main__':
    main()
