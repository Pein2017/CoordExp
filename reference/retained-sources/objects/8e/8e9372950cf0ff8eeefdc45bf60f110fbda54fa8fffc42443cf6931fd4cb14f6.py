"""Validate and flatten this one frozen paired physical review; no model work."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
REVIEW_ROOT = BASE/'fourth-fit-accepted-reviews-v3'
IMAGES = (25274, 59571, 99937, 210457, 219546, 323322, 351017, 388795, 417044, 477415, 528944)
PHASES = ((64, 'parent64', 'parent-step-00064', 'third-fit-v1'),
          (256, 'final256', 'fourth-step-00256', 'fourth-fit-v1'))


def read(path):
    return json.loads(Path(path).read_text())


def binding(path):
    path = Path(path)
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def verify_binding(value, expected=None):
    path = Path(value['path'])
    assert path.is_absolute(), path
    if expected is not None:
        assert path == expected, (path, expected)
    assert binding(path)['sha256'] == value['sha256'], path


def validate_one(review, packet, packet_path, native):
    verify_binding(review['source_packet'], packet_path)
    mp = HERE/'match-inheritance-v2.json'
    verify_binding(review['matching_inheritance'], mp)
    match_rule = next(r for r in read(mp)['rows'] if r['image_id'] == review['image_id'] and r['phase'] == review['phase'])
    inherited = {r['prediction_id']: r for r in match_rule['fixed_v4_matches'] + match_rule['v5_additional_matches']}
    for key, pk in [('target_catalog', 'target_catalog_v4'), ('annotation_catalog', 'annotation_catalog_v5')]:
        verify_binding(review[key], Path(packet['source'][pk]['path']))
    assert review['schema'] == 'fourth_fit_paired_owner_review.v1'
    assert review['status'] in ('candidate_ready', 'lead-accepted')
    assert review['image_id'] == packet['image_id']
    assert review['checkpoint_step'] == packet['checkpoint_step']
    raw = packet['full_raw_rows']
    decisions = review['decisions']
    assert review['raw_row_count'] == len(raw) == len(decisions)
    assert [(r['prediction_id'], r['generated_order']) for r in raw] == [
        (r['prediction_id'], r['generated_order']) for r in decisions]
    groups = {p: g for g in packet['rendered']['visual_groups'] for p in g['member_prediction_ids']}
    v4 = {str(r['owner_id']) for r in packet['target_catalog_references']}
    v5 = {str(r['owner_id']) for r in packet['annotation_catalog_v5_references']}
    assert v4 <= v5
    covered, annotated, seen = set(), set(), set()
    counts, evidence = Counter(), {}
    flattened = []
    for row, decision in zip(raw, decisions):
        pid = row['prediction_id']
        assert decision['visual_group_id'] == groups[pid]['visual_group_id'], pid
        physical = decision['physical_status']
        owner = decision['owner_id']
        if pid in inherited:
            assert owner == inherited[pid]['reference_owner_id'], (pid, 'inherited owner changed')
            assert decision['extent'] == 'reasonable', (pid, 'inherited match rejudged')
            assert decision['basis'] == 'iou_matched_inherited', pid
        assert physical in {'true_unique', 'repeat', 'false', 'unknown', 'invalid_output'}, pid
        assert decision['extent'] in {'reasonable', 'wrong', 'unknown'}, pid
        assert decision['class'] in {'verified', 'wrong', 'unknown'}, pid
        valid = row['status'] == 'parsed_valid'
        assert valid == (physical != 'invalid_output'), (pid, row['status'], physical)
        if physical in {'true_unique', 'repeat'}:
            assert owner is not None, (pid, 'identified physical row needs stable identity')
            assert isinstance(owner, str), pid
            assert physical == ('repeat' if owner in seen else 'true_unique'), (pid, owner, physical)
            seen.add(owner)
        qualified = valid and physical in {'true_unique', 'repeat'} and decision['extent'] == 'reasonable'
        assert decision['coverage_eligible'] == (qualified and owner in v4), pid
        assert decision['annotation_coverage_eligible'] == (qualified and owner in v5), pid
        if qualified and owner in v4:
            covered.add(owner)
        if qualified and owner in v5:
            annotated.add(owner)
        bbox_positive = valid and physical == 'true_unique' and decision['extent'] == 'reasonable' and owner in v5
        expected_ce = {'bbox': 'positive' if bbox_positive else 'mask',
                       'description': 'positive' if bbox_positive and decision['class'] == 'verified' else 'mask'}
        assert decision['direct_CE'] == expected_ce, (pid, decision['direct_CE'], expected_ce)
        assert decision['reason'].strip(), pid
        assert decision['evidence_paths'], pid
        for ep in decision['evidence_paths']:
            ep = ep['path'] if isinstance(ep, dict) else ep
            evidence[ep] = binding(ep)
        counts['physical_' + physical] += 1
        counts['extent_' + decision['extent']] += 1
        counts['class_' + decision['class']] += 1
        counts['bbox_positive'] += bbox_positive
        counts['description_positive'] += expected_ce['description'] == 'positive'
        counts['parser_drop_' + str(row.get('drop_reason'))] += not valid
        flattened.append({'image_id': review['image_id'], 'checkpoint_step': review['checkpoint_step'],
                          'phase': review['phase'], 'decision': decision, 'raw_row': row})
    summary = review['summary']
    for name, expected in [('target_owner_count', len(v4)), ('covered_owner_count', len(covered)),
                           ('missing_owner_count', len(v4 - covered)), ('annotation_target_owner_count', len(v5)),
                           ('physical_repeat_row_count', counts['physical_repeat']),
                           ('confirmed_false_row_count', counts['physical_false']),
                           ('physical_unknown_row_count', counts['physical_unknown']),
                           ('invalid_output_row_count', counts['physical_invalid_output']),
                           ('class_wrong_row_count', counts['class_wrong']),
                           ('class_unknown_row_count', counts['class_unknown'])]:
        assert summary[name] == expected, (review['image_id'], review['phase'], name, summary[name], expected)
    for name, expected in [('covered_owner_ids', covered), ('missing_owner_ids', v4 - covered),
                           ('annotation_covered_owner_ids', annotated), ('annotation_missing_owner_ids', v5 - annotated)]:
        assert set(summary[name]) == expected and len(summary[name]) == len(expected), name
    assert summary['raw_stop_reason'] == native['decode_stop_reason']
    cap = native['decode_stop_reason'] != 'im_end'
    assert bool(summary['cap_debt']) == cap
    return dict(image_id=review['image_id'], phase=review['phase'], checkpoint_step=review['checkpoint_step'],
                raw_rows=len(raw), target_owners=len(v4), covered_owner_ids=sorted(covered),
                missing_owner_ids=sorted(v4-covered), covered=len(covered), missing=len(v4-covered),
                annotation_target_owners=len(v5), annotation_covered_owner_ids=sorted(annotated),
                annotation_missing_owner_ids=sorted(v5-annotated), annotation_covered=len(annotated),
                annotation_missing=len(v5-annotated), counts=dict(counts), raw_stop_reason=summary['raw_stop_reason'],
                cap_debt=cap), flattened, evidence


def main():
    args = argparse.ArgumentParser()
    args.add_argument('--write', action='store_true')
    args = args.parse_args()
    per_image, flattened, sources, evidence, candidates = [], [], [], {}, []
    for image_id in IMAGES:
        pair = {}
        for step, phase, packet_dir, native_dir in PHASES:
            rp = REVIEW_ROOT / f'image-{image_id:012d}' / f'{phase}-review.json'
            pp = BASE / 'fourth-fit-eval-preparation-v1' / packet_dir / f'image-{image_id:012d}' / 'packet.json'
            np = BASE / native_dir / 'readback-recovery/rows' / f'step-{step:05d}-image-{image_id:012d}.json'
            review, packet, native = read(rp), read(pp), read(np)
            ledger, rows, ev = validate_one(review, packet, pp, native)
            pair[phase] = ledger
            flattened.extend(rows)
            evidence.update(ev)
            sources.extend([binding(rp), binding(pp), binding(np)])
            notes = Path(review['view_notes']['path'])
            verify_binding(review['view_notes'])
            assert notes.is_file() and notes.stat().st_size, notes
            sources.append(binding(notes))
            for candidate in review['new_owner_candidates']:
                candidates.append(dict(image_id=image_id, phase=phase, source_review=binding(rp), candidate=candidate))
            if phase == 'final256':
                before, after = set(pair['parent64']['covered_owner_ids']), set(ledger['covered_owner_ids'])
                for key, expected in [('retained_parent_owner_ids', before & after),
                                      ('lost_parent_owner_ids', before-after),
                                      ('newly_covered_fixed_owner_ids', after-before)]:
                    assert set(review['summary'][key]) == expected, (image_id, key)
                ledger.update(retained=len(before & after), lost=len(before-after), newly_covered=len(after-before),
                              retained_parent_owner_ids=sorted(before & after), lost_parent_owner_ids=sorted(before-after),
                              newly_covered_fixed_owner_ids=sorted(after-before))
            per_image.append(ledger)
    aggregate = {}
    for _, phase, _, _ in PHASES:
        rows = [r for r in per_image if r['phase'] == phase]
        counts = Counter()
        for r in rows:
            counts.update(r['counts'])
        aggregate[phase] = {k: sum(r.get(k, 0) for r in rows) for k in [
            'raw_rows', 'target_owners', 'covered', 'missing', 'annotation_target_owners',
            'annotation_covered', 'annotation_missing', 'retained', 'lost', 'newly_covered']}
        aggregate[phase].update(counts=dict(counts), eos_images=sum(not r['cap_debt'] for r in rows),
                                capped_images=sum(r['cap_debt'] for r in rows),
                                zero_missing_images=[r['image_id'] for r in rows if not r['missing']],
                                clean_complete_images=[r['image_id'] for r in rows if not r['missing'] and
                                    not r['cap_debt'] and not any(r['counts'].get(k,0) for k in (
                                        'physical_repeat','physical_false','physical_unknown','physical_invalid_output','extent_wrong','extent_unknown'))])
    result = {'schema': 'fourth_fit_paired_owner_acceptance.v1', 'status': 'candidate_ready',
              'scope': 'All 11 training images, cold natural greedy parent64 versus final256; fixed v4 primary and v5 supplemental. User v2: inherit IoU50 matches and visually adjudicate only unmatched.',
              'aggregate': aggregate, 'per_image': per_image, 'new_owner_candidate_events': len(candidates),
              'source_bindings': sources, 'producer': binding(Path(__file__)),
              'physical_conclusion_boundary': 'Mechanical validation does not replace root review of decision-bearing physical ambiguities.'}
    if args.write:
        out = BASE / 'fourth-fit-review-extraction-v1'
        out.mkdir(exist_ok=True)
        files = {'ledger.json': json.dumps(result, indent=2)+'\n',
                 'full-review-results.jsonl': ''.join(json.dumps(r)+'\n' for r in flattened),
                 'new-owner-candidates.jsonl': ''.join(json.dumps(r)+'\n' for r in candidates),
                 'evidence-index.json': json.dumps(list(evidence.values()), indent=2)+'\n'}
        for name, text in files.items():
            path = out/name
            if path.exists():
                assert path.read_text() == text, ('immutable output differs', path)
            else:
                path.write_text(text)
    print(json.dumps({'validated_images': len(IMAGES), 'aggregate': aggregate,
                      'new_owner_candidate_events': len(candidates), 'evidence_files': len(evidence)}, indent=2))


if __name__ == '__main__':
    main()
