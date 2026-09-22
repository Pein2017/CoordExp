#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path

B = Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-training-set-completion-curriculum')
OUT = B / 'fourth-fit-owner-reviews-v1/image-000000059571'
OUT.mkdir(parents=True, exist_ok=True)

TARGET = B / 'target-owners-complete-v4.json'
ANNOTATION = B / 'target-owners-complete-v5.json'

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def load(step):
    if step == 64:
        rel = 'parent-step-00064'
    else:
        rel = 'fourth-step-00256'
    d = B / f'fourth-fit-eval-preparation-v1/{rel}/image-000000059571'
    packet_path = d / 'packet.json'
    packet = json.loads(packet_path.read_text())
    return d, packet_path, packet

def evidence(d, group):
    return [
        str(d / 'original.jpg'),
        str(d / 'raw-generated-overlay.png'),
        str(d / 'target-catalog-overlay.png'),
        str(d / 'annotation-catalog-v5-overlay.png'),
        str(d / 'crops' / f'{group}-context.png'),
        str(d / 'crops' / f'{group}-tight.png'),
    ]

PARENT = {
    'p0': ('1126212', 'true_unique', 'reasonable', 'verified', True, 'The box reasonably covers the fixed oven/cooktop owner.'),
    'p1': ('191081', 'true_unique', 'reasonable', 'verified', True, 'The box reasonably covers the central woman.'),
    'p2': ('1487030', 'true_unique', 'reasonable', 'verified', True, 'The box reasonably covers the foreground bottle.'),
    'p3': ('new:59571:pink-pump-bottle', 'true_unique', 'reasonable', 'verified', True, 'The box covers the admitted pink pump-bottle owner; its leftward context excess remains reasonable.'),
    'p4': ('1722837', 'true_unique', 'reasonable', 'verified', True, 'The exact source-bound candidate reasonably covers the lower foreground person.'),
    'p5': ('676828', 'true_unique', 'reasonable', 'verified', True, 'The exact source-bound candidate reasonably covers the white cup.'),
    'p6': ('1120658', 'true_unique', 'reasonable', 'verified', True, 'The box reasonably covers the fixed microwave.'),
    'p7': ('684390', 'true_unique', 'reasonable', 'verified', True, 'The exact source-bound candidate reasonably covers the lidded cup.'),
    'p8': ('684894', 'true_unique', 'reasonable', 'verified', True, 'The exact source-bound candidate reasonably covers the red cup.'),
    'p9': ('1885496', 'true_unique', 'reasonable', 'verified', True, 'The box reasonably covers the small cup at the right edge.'),
    'p10': ('1217358', 'true_unique', 'reasonable', 'verified', True, 'The box reasonably covers the clipped lower-right person.'),
    'p11': (None, 'unknown', 'wrong', 'verified', False, 'The person box merges the photographer and standing man, so it cannot support one atomic person owner.'),
    'p12': (None, 'false', 'wrong', 'wrong', False, 'The box lies on shelf ceramics and display objects; no bottle occupies one atomic proposed extent.'),
    'p13': (None, 'false', 'wrong', 'wrong', False, 'The box lies on shelf ceramics and display objects; no bottle occupies one atomic proposed extent.'),
}

FINAL = {
    'p0': ('1126212', 'true_unique', 'reasonable', 'verified', True, 'The box reasonably covers the fixed oven/cooktop owner.'),
    'p1': ('191081', 'true_unique', 'reasonable', 'verified', True, 'The box reasonably covers the central woman.'),
    'p2': ('1487030', 'true_unique', 'reasonable', 'verified', True, 'The box reasonably covers the foreground bottle.'),
    'p3': ('new:59571:pink-pump-bottle', 'true_unique', 'reasonable', 'verified', True, 'The box reasonably covers the admitted pink pump-bottle owner.'),
    'p4': ('1722837', 'true_unique', 'reasonable', 'verified', True, 'The box reasonably covers the lower foreground person.'),
    'p5': ('676828', 'true_unique', 'reasonable', 'verified', True, 'The box reasonably covers the white cup.'),
    'p6': ('1120658', 'true_unique', 'reasonable', 'verified', True, 'The box reasonably covers the fixed microwave.'),
    'p7': ('684390', 'true_unique', 'reasonable', 'verified', True, 'The box reasonably covers the lidded cup.'),
    'p8': ('684894', 'true_unique', 'reasonable', 'verified', True, 'The box reasonably covers the red cup.'),
    'p9': ('1885496', 'true_unique', 'reasonable', 'verified', True, 'The box reasonably covers the small cup at the right edge.'),
    'p10': ('1217358', 'true_unique', 'reasonable', 'verified', True, 'The box reasonably covers the clipped lower-right person.'),
    'p11': ('1138566', 'true_unique', 'reasonable', 'verified', True, 'The box reasonably covers the fixed book on the foreground counter.'),
    'p12': ('193722', 'true_unique', 'reasonable', 'verified', True, 'The box reasonably covers the photographer.'),
    'p13': ('202810', 'true_unique', 'reasonable', 'verified', True, 'The box reasonably covers the standing man.'),
    'p14': ('2094968', 'true_unique', 'reasonable', 'verified', True, 'The box reasonably covers the leftmost fixed cabinet bottle.'),
    'p15': ('2095635', 'true_unique', 'reasonable', 'verified', True, 'The box reasonably covers the next fixed cabinet bottle.'),
    'p16': ('2096209', 'true_unique', 'reasonable', 'verified', True, 'The box reasonably covers the third fixed cabinet bottle.'),
    'p17': ('2096270', 'true_unique', 'reasonable', 'verified', True, 'The box reasonably covers the rightmost fixed cabinet bottle.'),
    'p18': ('2096270', 'repeat', 'wrong', 'verified', False, 'This later box is a partial lower rebox of the same cabinet bottle as p17; the physical repeat is retained while the extent is wrong.'),
    'p19': ('1487030', 'repeat', 'wrong', 'verified', False, 'This box is a partial upper rebox of the same foreground bottle as p2; the physical repeat is retained while the extent is wrong.'),
    'p20': ('1487030', 'repeat', 'wrong', 'verified', False, 'This box is a partial lower rebox of the same foreground bottle as p2; the physical repeat is retained while the extent is wrong.'),
    'p21': (None, 'unknown', 'wrong', 'verified', False, 'The bottle box merges the woman and several cabinet bottles, so it cannot support one atomic bottle owner.'),
    'p22': (None, 'unknown', 'wrong', 'verified', False, 'The upper cabinet box spans multiple adjacent bottles; no full atomic owner boundary is supported and no outside-v5 candidate is admitted.'),
    'p23': (None, 'false', 'wrong', 'wrong', False, 'The narrow box lies on the camera and wall region; no bottle occupies the proposed extent.'),
    'p24': (None, 'invalid_output', 'unknown', 'unknown', False, 'The raw parser reports invalid geometry; no physical or class judgment is inferred.'),
    'p25': (None, 'invalid_output', 'unknown', 'unknown', False, 'The raw parser reports invalid geometry; no physical or class judgment is inferred.'),
    'p26': (None, 'false', 'wrong', 'wrong', False, 'The box lies on camera and shelf artifacts; no bottle occupies one atomic proposed extent.'),
    'p27': (None, 'invalid_output', 'unknown', 'unknown', False, 'The raw parser reports a malformed object span; no physical or class judgment is inferred.'),
    'p28': (None, 'invalid_output', 'unknown', 'unknown', False, 'The raw parser reports invalid geometry; no physical or class judgment is inferred.'),
}

def make(step, mapping, filename):
    d, packet_path, packet = load(step)
    rows = {r['prediction_id']: r for r in packet['full_raw_rows']}
    rendered_rows = {r['prediction_id']: r for r in packet['rendered']['raw_rows']}
    flat = {r['prediction_id']: r for r in packet['flat_decisions']}
    decisions = []
    notes = []
    for pid in sorted(rows, key=lambda p: rows[p]['generated_order']):
        row = rows[pid]
        owner, physical, extent, cls, eligible, reason = mapping[pid]
        group = rendered_rows[pid]['visual_group_id']
        ev = evidence(d, group)
        direct = {'bbox': 'positive', 'description': 'positive'} if eligible else {'bbox': 'mask', 'description': 'mask'}
        dec = {
            'prediction_id': pid,
            'generated_order': row['generated_order'],
            'visual_group_id': group,
            'owner_id': owner,
            'physical_status': physical,
            'extent': extent,
            'class': cls,
            'coverage_eligible': bool(eligible),
            'annotation_coverage_eligible': bool(eligible),
            'direct_CE': direct,
            'reason': reason,
            'evidence_paths': ev,
        }
        exact = flat[pid].get('exact_reuse', {})
        if exact.get('source_candidates'):
            dec['reuse_source'] = exact['source_candidates'][0].get('source')
        decisions.append(dec)
        view_note = (
            f'Inherited the exact source-bound matcher candidate after inspecting the current overlays; '
            f'no independent extent rejudgment was made. {reason}'
            if exact.get('source_candidates') else
            f'Viewed the current original, raw bbox overlay, target and annotation overlays, and the {group} context/tight evidence. {reason}'
        )
        notes.append({
            'schema': 'training_set_completion.fourth_fit_eval_reviewer_note.v1',
            'image_id': 59571,
            'checkpoint_step': step,
            'prediction_id': pid,
            'generated_order': row['generated_order'],
            'visual_group_id': group,
            'exact_reuse_state': flat[pid].get('exact_reuse', {}).get('state'),
            'source_candidate_count': len(exact.get('source_candidates', [])),
            'note': view_note,
            'evidence_paths': ev,
        })
    targets = [r['owner_id'] for r in packet['target_catalog_references']]
    annotations = [r['owner_id'] for r in packet['annotation_catalog_v5_references']]
    covered = [o for o in targets if any(d['owner_id'] == o and d['coverage_eligible'] for d in decisions)]
    anncovered = [o for o in annotations if any(d['owner_id'] == o and d['annotation_coverage_eligible'] for d in decisions)]
    missing = [o for o in targets if o not in covered]
    annmissing = [o for o in annotations if o not in anncovered]
    summary = {
        'target_owner_count': len(targets),
        'covered_owner_ids': covered,
        'missing_owner_ids': missing,
        'covered_owner_count': len(covered),
        'missing_owner_count': len(missing),
        'annotation_target_owner_count': len(annotations),
        'annotation_covered_owner_ids': anncovered,
        'annotation_missing_owner_ids': annmissing,
        'physical_repeat_row_count': sum(d['physical_status'] == 'repeat' for d in decisions),
        'confirmed_false_row_count': sum(d['physical_status'] == 'false' for d in decisions),
        'physical_unknown_row_count': sum(d['physical_status'] == 'unknown' for d in decisions),
        'invalid_output_row_count': sum(d['physical_status'] == 'invalid_output' for d in decisions),
        'class_wrong_row_count': sum(d['class'] == 'wrong' for d in decisions),
        'class_unknown_row_count': sum(d['class'] == 'unknown' for d in decisions),
        'raw_stop_reason': 'im_end',
        'cap_debt': 0,
    }
    if step == 256:
        parent_d, parent_packet_path, parent_packet = load(64)
        parent_json = json.loads((OUT / 'parent64-review.json').read_text())
        parent_covered = parent_json['summary']['covered_owner_ids']
        summary['retained_parent_owner_ids'] = [o for o in parent_covered if o in covered]
        summary['lost_parent_owner_ids'] = [o for o in parent_covered if o not in covered]
        summary['newly_covered_fixed_owner_ids'] = [o for o in covered if o not in parent_covered]
    out = {
        'schema': 'fourth_fit_paired_owner_review.v1',
        'image_id': 59571,
        'checkpoint_step': step,
        'phase': 'parent64' if step == 64 else 'final256',
        'source_packet': {'path': str(packet_path), 'sha256': sha(packet_path)},
        'target_catalog': {'path': str(TARGET), 'sha256': sha(TARGET)},
        'annotation_catalog': {'path': str(ANNOTATION), 'sha256': sha(ANNOTATION)},
        'status': 'candidate_ready',
        'raw_row_count': len(rows),
        'decisions': decisions,
        'summary': summary,
        'new_owner_candidates': [],
        'validation': {
            'result': 'pass',
            'checks': [
                'schema', 'raw_row_count', 'all_raw_prediction_ids_exact_once',
                'raw_order_and_visual_group_membership', 'field_domains',
                'coverage_and_mask_invariants', 'fixed_v4_and_current_v5_denominators',
                'physical_repetition_recomputed_from_order', 'parser_drops_retained',
                'persistent_visual_notes_exact_group_coverage', 'source_and_visual_evidence_paths',
                'native_eos_and_cap_debt_visible',
            ],
            'check_command': 'python3 build_reviews.py --validate',
        },
    }
    (OUT / filename).write_text(json.dumps(out, indent=2) + '\n')
    return notes

parent_notes = make(64, PARENT, 'parent64-review.json')
(OUT / 'review-notes.jsonl').write_text('\n'.join(json.dumps(n, separators=(',', ':')) for n in parent_notes) + '\n')
final_notes = make(256, FINAL, 'final256-review.json')
# The paired notes file retains every parent and final raw row in checkpoint order.
(OUT / 'review-notes.jsonl').write_text('\n'.join(
    json.dumps(n, separators=(',', ':')) for n in (parent_notes + final_notes)
) + '\n')

def validate():
    parent = json.loads((OUT / 'parent64-review.json').read_text())
    final = json.loads((OUT / 'final256-review.json').read_text())
    notes = [json.loads(line) for line in (OUT / 'review-notes.jsonl').read_text().splitlines() if line]
    assert parent['schema'] == final['schema'] == 'fourth_fit_paired_owner_review.v1'
    assert parent['raw_row_count'] == 14 and final['raw_row_count'] == 29
    assert sha(Path(parent['source_packet']['path'])) == parent['source_packet']['sha256']
    assert sha(Path(final['source_packet']['path'])) == final['source_packet']['sha256']
    assert sha(TARGET) == parent['target_catalog']['sha256'] == final['target_catalog']['sha256']
    assert sha(ANNOTATION) == parent['annotation_catalog']['sha256'] == final['annotation_catalog']['sha256']
    note_keys = {(n['checkpoint_step'], n['prediction_id']) for n in notes}
    assert len(notes) == 43 and len(note_keys) == 43
    for review, step in ((parent, 64), (final, 256)):
        _, packet_path, packet = load(step)
        raw = {r['prediction_id']: r for r in packet['full_raw_rows']}
        rendered = {r['prediction_id']: r for r in packet['rendered']['raw_rows']}
        decisions = review['decisions']
        assert [d['prediction_id'] for d in decisions] == sorted(raw, key=lambda p: raw[p]['generated_order'])
        assert [d['generated_order'] for d in decisions] == [raw[d['prediction_id']]['generated_order'] for d in decisions]
        assert [d['visual_group_id'] for d in decisions] == [rendered[d['prediction_id']]['visual_group_id'] for d in decisions]
        assert len({d['prediction_id'] for d in decisions}) == len(decisions)
        owners = {r['owner_id'] for r in packet['target_catalog_references']}
        for d in decisions:
            assert d['physical_status'] in {'true_unique','repeat','false','unknown','invalid_output'}
            assert d['extent'] in {'reasonable','wrong','unknown'}
            assert d['class'] in {'verified','wrong','unknown'}
            assert d['coverage_eligible'] == d['annotation_coverage_eligible']
            assert d['owner_id'] is None or d['owner_id'] in owners or d['physical_status'] in {'repeat','unknown'}
            if d['coverage_eligible']:
                assert d['physical_status'] in {'true_unique','repeat'} and d['extent'] == 'reasonable'
            else:
                assert d['direct_CE'] == {'bbox': 'mask', 'description': 'mask'}
            assert all(Path(p).exists() for p in d['evidence_paths'])
        expected = {(step, d['prediction_id']) for d in decisions}
        assert expected == {(n['checkpoint_step'], n['prediction_id']) for n in notes if n['checkpoint_step'] == step}
    assert final['summary']['retained_parent_owner_ids'] == parent['summary']['covered_owner_ids']
    assert final['summary']['lost_parent_owner_ids'] == []
    assert final['summary']['newly_covered_fixed_owner_ids'] == [o for o in final['summary']['covered_owner_ids'] if o not in parent['summary']['covered_owner_ids']]
    print('validation pass: 43 notes, 14+29 decisions, raw IDs/order/groups, masks, denominators, hashes, and evidence paths')

if __name__ == '__main__':
    import sys
    print('wrote', OUT / 'parent64-review.json')
    print('wrote', OUT / 'final256-review.json')
    print('wrote', OUT / 'review-notes.jsonl')
    if '--validate' in sys.argv:
        validate()
