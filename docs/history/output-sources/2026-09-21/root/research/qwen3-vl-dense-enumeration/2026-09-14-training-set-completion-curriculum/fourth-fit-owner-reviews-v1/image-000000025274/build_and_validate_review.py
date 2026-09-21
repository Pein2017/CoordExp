#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path


BASE = Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-training-set-completion-curriculum')
OUT = BASE / 'fourth-fit-owner-reviews-v1/image-000000025274'
PREP = BASE / 'fourth-fit-eval-preparation-v1'
V4 = BASE / 'target-owners-complete-v4.json'
V5 = BASE / 'target-owners-complete-v5.json'
MATCH_INHERITANCE = BASE / 'fourth-fit-owner-reviews-v1/match-inheritance-v2.json'
IMAGE_ID = 25274
SCHEMA = 'fourth_fit_paired_owner_review.v1'


def read(path):
    with path.open() as handle:
        return json.load(handle)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def binding(path):
    return {'path': str(path), 'sha256': sha256(path)}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def phase_paths(phase):
    directory = PREP / phase / 'image-000000025274'
    return {
        'directory': directory,
        'packet': directory / 'packet.json',
        'original': directory / 'original.jpg',
        'raw_overlay': directory / 'raw-generated-overlay.png',
        'v4_overlay': directory / 'target-catalog-overlay.png',
        'v5_overlay': directory / 'annotation-catalog-v5-overlay.png',
    }


def review_notes():
    shared = phase_paths('parent-step-00064')
    notes = [
        {
            'phase': 'shared',
            'view_id': 'original',
            'evidence_path': str(shared['original']),
            'covered_visual_group_ids': [],
            'interpretation': 'Dense Shibuya crossing scene. The foreground line contains many distinct people plus three traffic lights, one bus, one backpack, and two small handbags. Physical owner identity is assessed from the full scene and current full-reference boxes; overlapping person boxes are not treated as owner identity by themselves.',
        },
        {
            'phase': 'shared',
            'view_id': 'fixed-v4-overlay',
            'evidence_path': str(shared['v4_overlay']),
            'covered_visual_group_ids': [],
            'interpretation': 'Fixed-v4 has 33 image-local owners. It includes the disambiguated brown-jacket L08, patterned-skirt L09, and neighboring green-coat GT1323904 as three separate owners, but excludes the later orange-skirt admission.',
        },
        {
            'phase': 'shared',
            'view_id': 'annotation-v5-overlay',
            'evidence_path': str(shared['v5_overlay']),
            'covered_visual_group_ids': [],
            'interpretation': 'Current-v5 has 34 image-local owners. It preserves all fixed-v4 owners and adds the root-reviewed full orange-skirt person reference third-fit:new:25274:orange-skirt-person at [805,659,832,795].',
        },
        {
            'phase': 'shared',
            'view_id': 'l08-l09-green-reference-crop',
            'evidence_path': str(BASE / 'second-fit-root-rulings-v1/image-000000025274-three-owner-refs-crop.png'),
            'covered_visual_group_ids': [],
            'interpretation': 'The reference crop confirms three physical people: green-coat GT1323904 at left, brown-jacket L08 in the middle with full bins [341,640,371,798], and dark-top patterned-skirt L09 at right with full bins [359,651,395,798]. The initial visual reading treated parent g0008 as too far left; that extent judgment is superseded by match-inheritance-v2, which deterministically matches p8 to L08 at IoU 0.7898812596799174 and assigns reasonable extent.',
        },
        {
            'phase': 'shared',
            'view_id': 'orange-skirt-full-reference',
            'evidence_path': str(BASE / 'third-fit-root-rulings-v1/image25274-orange-skirt-reference.png'),
            'covered_visual_group_ids': [],
            'interpretation': 'The full reference isolates the v5-only orange-skirt person between L13 and GT1331590. No final256 prediction covers this full reference, so it remains an annotation-v5 miss and is not a new candidate.',
        },
    ]
    groups = [
        ('parent64', 'parent-step-00064', 0, 5, 'g0000-g0005 each shows one distinct person. L02, the white-coat person, L04, L05, L06, and L07 have reasonable visible full extents and verified person descriptions.'),
        ('parent64', 'parent-step-00064', 6, 11, 'Initial visual interpretation: g0006=GT1330249, g0007=GT1326797, g0009=L09, g0010=GT2159701, and g0011=GT1212609 appeared reasonable; g0008 appeared left-shifted into green-coat GT1323904. All six are matcher-owned under match-inheritance-v2, so the visual extent judgments are preserved here only as superseded notes; p8 inherits L08 with reasonable extent at IoU 0.7898812596799174.'),
        ('parent64', 'parent-step-00064', 12, 17, 'g0012=GT1275404, g0013=L10, g0014=GT1383876 traffic light, g0015=GT407912 traffic light, g0016=GT408513 traffic light, and g0017=L11 are six distinct owners with reasonable extent and verified descriptions.'),
        ('parent64', 'parent-step-00064', 18, 21, 'Initial visual interpretation: matched g0018=L13, g0019=GT1714596, and g0021=GT1296103 appeared distinct; those judgments are superseded by match-inheritance-v2. Unmatched g0020 straddles GT1331684 and L11 and carries a malformed literal description, so physical owner/extent are unknown and class is wrong; it is masked rather than counted false.'),
        ('final256', 'fourth-step-00256', 0, 5, 'g0000-g0005 each isolates its current-v5 person owner with reasonable full extent: L02, white-coat person, L04, L05, L06, and L07.'),
        ('final256', 'fourth-step-00256', 6, 11, 'g0006=GT1330249, g0007=GT1326797, g0008=L08, g0009=L09, g0010=GT2159701, and g0011=GT1212609. The corrected L08/L09 boxes match their full references and stay distinct from green-coat GT1323904.'),
        ('final256', 'fourth-step-00256', 12, 17, 'g0012=GT1275404, g0013=L10, three traffic-light groups g0014-g0016, and g0017=L11 are six distinct reasonable boxes with verified descriptions.'),
        ('final256', 'fourth-step-00256', 18, 23, 'g0018=L13, g0019=GT1331590, g0020=L15, g0021=GT1714596, g0022=GT1296103, and g0023=GT1331684 are six distinct reasonable person boxes. None is the v5-only orange-skirt person located between L13 and GT1331590.'),
        ('final256', 'fourth-step-00256', 24, 29, 'g0024=L16 person, g0025=GT1166904 backpack, g0026=GT1308843 person, g0027=green-coat GT1323904 person, g0028=GT1365130 bus, and g0029=GT2011786 person all match one full current reference with reasonable extent.'),
        ('final256', 'fourth-step-00256', 30, 32, 'g0030 is the black-coated person; g0031 and g0032 are two distinct small handbags at L19 and L20. Geometry is reasonable and the visible classes match the generated literals.'),
    ]
    for phase, packet_phase, first, last, interpretation in groups:
        path = OUT / 'evidence' / f'{packet_phase}-groups-{first:02d}-{last:02d}.png'
        notes.append({
            'phase': phase,
            'view_id': f'groups-{first:02d}-{last:02d}-context-tight-sheet',
            'evidence_path': str(path),
            'covered_visual_group_ids': [f'g{i:04d}' for i in range(first, last + 1)],
            'interpretation': interpretation,
        })
    for phase, packet_phase, count, interpretation in [
        ('parent64', 'parent-step-00064', 22, 'Parent raw overlay confirms 22 parser-valid boxes followed by malformed/collapsed raw geometry at the length cap. No parser-dropped row supports a physical or class conclusion.'),
        ('final256', 'fourth-step-00256', 33, 'Final raw overlay confirms 33 parser-valid boxes, one per fixed-v4 owner, with no repeated physical owner. The route ends natively after the two handbag rows.'),
    ]:
        path = phase_paths(packet_phase)['raw_overlay']
        notes.append({
            'phase': phase,
            'view_id': 'raw-generated-overlay',
            'evidence_path': str(path),
            'covered_visual_group_ids': [f'g{i:04d}' for i in range(count)],
            'interpretation': interpretation,
        })
    enriched = []
    for note in notes:
        path = Path(note['evidence_path'])
        enriched.append({
            'schema': 'fourth_fit_paired_owner_review_note.v1',
            'image_id': IMAGE_ID,
            **note,
            'evidence_sha256': sha256(path),
            'decision_use': 'Observation ledger only for matcher-owned rows; match-inheritance-v2 governs their owner and extent. Human judgment remains active only for unmatched parent p20.',
        })
    return enriched


def inheritance_row(phase_name):
    rows = [row for row in read(MATCH_INHERITANCE)['rows'] if row['image_id'] == IMAGE_ID and row['phase'] == phase_name]
    assert len(rows) == 1
    return rows[0]


def selector_image_row(path):
    rows = []
    def visit(value):
        if isinstance(value, dict):
            if value.get('image_id') == IMAGE_ID and 'cap_debt' in value:
                rows.append(value)
            for nested in value.values():
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)
    visit(read(path))
    assert len(rows) == 1
    return rows[0]


def make_decisions(packet, phase, v4_ids, v5_ids):
    rendered = {row['prediction_id']: row for row in packet['rendered']['raw_rows']}
    flat = {row['prediction_id']: row for row in packet['flat_decisions']}
    phase_name = 'parent64' if phase == 'parent-step-00064' else 'final256'
    inherited = inheritance_row(phase_name)
    matches = {}
    for catalog, key in [('fixed_v4', 'fixed_v4_matches'), ('v5_additional', 'v5_additional_matches')]:
        for match in inherited[key]:
            matches[match['prediction_id']] = {**match, 'catalog': catalog}
    assert set(inherited['unmatched_valid_prediction_ids']).isdisjoint(matches)
    v5_categories = {row['owner_id']: row.get('category') for row in packet['annotation_catalog_v5_references']}
    decisions = []
    for row in packet['full_raw_rows']:
        prediction_id = row['prediction_id']
        rendered_row = rendered[prediction_id]
        group_id = rendered_row['visual_group_id']
        if row['status'] != 'parsed_valid':
            decisions.append({
                'prediction_id': prediction_id,
                'generated_order': row['generated_order'],
                'visual_group_id': group_id,
                'owner_id': None,
                'physical_status': 'invalid_output',
                'extent': 'unknown',
                'class': 'unknown',
                'coverage_eligible': False,
                'annotation_coverage_eligible': False,
                'direct_CE': {'bbox': 'mask', 'description': 'mask'},
                'basis': 'parser_invalid',
                'reason': f"Parser-dropped raw row ({row['drop_reason']}); no physical owner, extent, or class conclusion is made.",
                'evidence_paths': [str(phase_paths(phase)['packet']), str(phase_paths(phase)['raw_overlay'])],
            })
            continue
        order = row['generated_order']
        match = matches.get(prediction_id)
        reuse = flat[prediction_id]['exact_reuse']
        if match is None:
            assert prediction_id in inherited['unmatched_valid_prediction_ids']
            assert phase_name == 'parent64' and prediction_id == 'p20'
            owner_id, extent, class_status = None, 'unknown', 'wrong'
            physical_status = 'unknown'
            basis = 'human_review_unmatched'
        else:
            owner_id = match['reference_owner_id']
            extent = 'reasonable'
            physical_status = 'true_unique'
            basis = 'iou_matched_inherited'
            catalog_category = v5_categories[owner_id]
            if catalog_category is not None and catalog_category == row['description']:
                class_status = 'verified'
            elif reuse['state'] == 'exact_literal_reuse_candidate' and reuse['source_candidates'][0]['semantic']['class'] == 'verified':
                class_status = 'verified'
            else:
                class_status = 'unknown'
        coverage = physical_status in {'true_unique', 'repeat'} and extent == 'reasonable' and owner_id in v4_ids
        annotation_coverage = physical_status in {'true_unique', 'repeat'} and extent == 'reasonable' and owner_id in v5_ids
        positive = physical_status == 'true_unique' and extent == 'reasonable' and owner_id in v5_ids
        if match is None:
            evidence = [
                str(phase_paths(phase)['original']),
                str(phase_paths(phase)['raw_overlay']),
                str(phase_paths(phase)['v5_overlay']),
                rendered_row['shared_crop_paths']['context'],
                rendered_row['shared_crop_paths']['tight'],
            ]
            reason = 'The box spans portions of GT1331684 and L11, and the literal description is malformed (person plus control-token text); owner and extent remain unknown, class is wrong, and the row is masked rather than counted false.'
        else:
            evidence = [str(MATCH_INHERITANCE), str(phase_paths(phase)['packet']), str(V4), str(V5)]
            class_reason = 'catalog category exactly matches the literal' if class_status == 'verified' and v5_categories[owner_id] is not None else ('packet-provided exact source-bound prior evidence verifies the class' if class_status == 'verified' else 'catalog category is unset and no exact source-bound class proof applies, so class remains unknown')
            reason = f"match-inheritance-v2 {match['catalog']} one-to-one IoU-0.5 match assigns owner {owner_id} at IoU {match['iou']} with extent reasonable; {class_reason}. Physical repetition is recomputed from generated order."
        decision = {
            'prediction_id': prediction_id,
            'generated_order': order,
            'visual_group_id': group_id,
            'owner_id': owner_id,
            'physical_status': physical_status,
            'extent': extent,
            'class': class_status,
            'coverage_eligible': coverage,
            'annotation_coverage_eligible': annotation_coverage,
            'direct_CE': {
                'bbox': 'positive' if positive else 'mask',
                'description': 'positive' if positive and class_status == 'verified' else 'mask',
            },
            'basis': basis,
            'reason': reason,
            'evidence_paths': evidence,
        }
        if reuse['state'] == 'exact_literal_reuse_candidate':
            decision['reuse_source'] = reuse['source_candidates'][0]['source']
        if match is not None:
            decision['match_catalog'] = match['catalog']
            decision['match_iou'] = match['iou']
        decisions.append(decision)
    return decisions


def make_review(phase, step, output_name, retained_parent=None):
    paths = phase_paths(phase)
    packet = read(paths['packet'])
    v4_ids = [row['owner_id'] for row in packet['target_catalog_references']]
    v5_ids = [row['owner_id'] for row in packet['annotation_catalog_v5_references']]
    decisions = make_decisions(packet, phase, set(v4_ids), set(v5_ids))
    covered_set = {d['owner_id'] for d in decisions if d['coverage_eligible']}
    annotation_covered_set = {d['owner_id'] for d in decisions if d['annotation_coverage_eligible']}
    covered = [owner_id for owner_id in v4_ids if owner_id in covered_set]
    missing = [owner_id for owner_id in v4_ids if owner_id not in covered_set]
    annotation_covered = [owner_id for owner_id in v5_ids if owner_id in annotation_covered_set]
    annotation_missing = [owner_id for owner_id in v5_ids if owner_id not in annotation_covered_set]
    selector_path = Path(packet['source']['scored_selector']['path'])
    selector_row = selector_image_row(selector_path)
    summary = {
        'target_owner_count': len(v4_ids),
        'covered_owner_ids': covered,
        'missing_owner_ids': missing,
        'covered_owner_count': len(covered),
        'missing_owner_count': len(missing),
        'annotation_target_owner_count': len(v5_ids),
        'annotation_covered_owner_ids': annotation_covered,
        'annotation_missing_owner_ids': annotation_missing,
        'physical_repeat_row_count': sum(d['physical_status'] == 'repeat' for d in decisions),
        'confirmed_false_row_count': sum(d['physical_status'] == 'false' for d in decisions),
        'physical_unknown_row_count': sum(d['physical_status'] == 'unknown' for d in decisions),
        'invalid_output_row_count': sum(d['physical_status'] == 'invalid_output' for d in decisions),
        'class_wrong_row_count': sum(d['class'] == 'wrong' for d in decisions),
        'class_unknown_row_count': sum(d['class'] == 'unknown' for d in decisions),
        'raw_stop_reason': selector_row['decode_stop_reason'],
        'cap_debt': selector_row['cap_debt'],
    }
    if retained_parent is not None:
        parent_ids = retained_parent['summary']['covered_owner_ids']
        summary.update({
            'retained_parent_owner_ids': [owner_id for owner_id in parent_ids if owner_id in covered_set],
            'lost_parent_owner_ids': [owner_id for owner_id in parent_ids if owner_id not in covered_set],
            'newly_covered_fixed_owner_ids': [owner_id for owner_id in v4_ids if owner_id in covered_set and owner_id not in set(parent_ids)],
        })
    review = {
        'schema': SCHEMA,
        'image_id': IMAGE_ID,
        'checkpoint_step': step,
        'phase': 'parent64' if step == 64 else 'final256',
        'source_packet': binding(paths['packet']),
        'target_catalog': binding(V4),
        'annotation_catalog': binding(V5),
        'matching_inheritance': binding(MATCH_INHERITANCE),
        'status': 'candidate_ready',
        'raw_row_count': len(packet['full_raw_rows']),
        'decisions': decisions,
        'summary': summary,
        'new_owner_candidates': [],
        'validation': {
            'all_raw_rows_decided': True,
            'raw_ids_orders_and_groups_match_packet': True,
            'source_hashes_verified': True,
            'fixed_v4_partition_verified': True,
            'annotation_v5_partition_verified': True,
            'direct_ce_masks_verified': True,
            'native_eos_and_cap_verified': True,
            'viewed_evidence_verified': True,
            'fixed_v4_global_owner_count': read(V4)['atomic_target_count'],
            'current_v5_global_owner_count': read(V5)['atomic_target_count'],
        },
    }
    write_json(OUT / output_name, review)
    return review


def validate_review(path, expected_phase, expected_step, notes):
    review = read(path)
    assert review['schema'] == SCHEMA
    assert review['status'] == 'candidate_ready'
    assert review['image_id'] == IMAGE_ID
    assert review['phase'] == expected_phase and review['checkpoint_step'] == expected_step
    packet = read(Path(review['source_packet']['path']))
    assert review['source_packet']['sha256'] == sha256(Path(review['source_packet']['path']))
    assert review['target_catalog'] == binding(V4)
    assert review['annotation_catalog'] == binding(V5)
    assert review['matching_inheritance'] == binding(MATCH_INHERITANCE)
    assert read(V4)['atomic_target_count'] == 232
    assert read(V5)['atomic_target_count'] == 246
    selector_binding = packet['source']['scored_selector']
    selector_path = Path(selector_binding['path'])
    assert selector_binding['sha256'] == sha256(selector_path)
    selector_row = selector_image_row(selector_path)
    assert review['summary']['raw_stop_reason'] == selector_row['decode_stop_reason']
    assert review['summary']['cap_debt'] == selector_row['cap_debt']
    assert review['raw_row_count'] == selector_row['raw_row_count']
    assert sum(row['status'] == 'parsed_valid' for row in packet['full_raw_rows']) == selector_row['valid_prediction_count']
    assert (review['summary']['raw_stop_reason'] == 'im_end') == selector_row['natural_eos']
    rows = packet['full_raw_rows']
    rendered = {row['prediction_id']: row for row in packet['rendered']['raw_rows']}
    decisions = review['decisions']
    inherited = inheritance_row(expected_phase)
    inherited_matches = {
        match['prediction_id']: match
        for key in ('fixed_v4_matches', 'v5_additional_matches')
        for match in inherited[key]
    }
    assert len(decisions) == len(rows) == review['raw_row_count']
    for row, decision in zip(rows, decisions):
        assert decision['prediction_id'] == row['prediction_id']
        assert decision['generated_order'] == row['generated_order']
        assert decision['visual_group_id'] == rendered[row['prediction_id']]['visual_group_id']
        assert decision['physical_status'] in {'true_unique', 'repeat', 'false', 'unknown', 'invalid_output'}
        assert decision['extent'] in {'reasonable', 'wrong', 'unknown'}
        assert decision['class'] in {'verified', 'wrong', 'unknown'}
        assert all(Path(evidence).exists() for evidence in decision['evidence_paths'])
        if row['status'] != 'parsed_valid':
            assert decision['physical_status'] == 'invalid_output'
            assert decision['basis'] == 'parser_invalid'
        elif row['prediction_id'] in inherited_matches:
            match = inherited_matches[row['prediction_id']]
            assert decision['basis'] == 'iou_matched_inherited'
            assert decision['owner_id'] == match['reference_owner_id']
            assert decision['extent'] == 'reasonable'
            assert decision['match_iou'] == match['iou']
        else:
            assert row['prediction_id'] in inherited['unmatched_valid_prediction_ids']
            assert decision['basis'] == 'human_review_unmatched'
        eligible_physical = row['status'] == 'parsed_valid' and decision['physical_status'] in {'true_unique', 'repeat'} and decision['extent'] == 'reasonable'
        v4_ids = {item['owner_id'] for item in packet['target_catalog_references']}
        v5_ids = {item['owner_id'] for item in packet['annotation_catalog_v5_references']}
        assert decision['coverage_eligible'] == (eligible_physical and decision['owner_id'] in v4_ids)
        assert decision['annotation_coverage_eligible'] == (eligible_physical and decision['owner_id'] in v5_ids)
        positive = row['status'] == 'parsed_valid' and decision['physical_status'] == 'true_unique' and decision['extent'] == 'reasonable' and decision['owner_id'] in v5_ids
        assert decision['direct_CE']['bbox'] == ('positive' if positive else 'mask')
        assert decision['direct_CE']['description'] == ('positive' if positive and decision['class'] == 'verified' else 'mask')
        if decision['physical_status'] == 'repeat':
            assert decision['direct_CE'] == {'bbox': 'mask', 'description': 'mask'}
    summary = review['summary']
    covered = {d['owner_id'] for d in decisions if d['coverage_eligible']}
    annotation_covered = {d['owner_id'] for d in decisions if d['annotation_coverage_eligible']}
    v4_order = [row['owner_id'] for row in packet['target_catalog_references']]
    v5_order = [row['owner_id'] for row in packet['annotation_catalog_v5_references']]
    assert summary['covered_owner_ids'] == [owner for owner in v4_order if owner in covered]
    assert summary['missing_owner_ids'] == [owner for owner in v4_order if owner not in covered]
    assert summary['annotation_covered_owner_ids'] == [owner for owner in v5_order if owner in annotation_covered]
    assert summary['annotation_missing_owner_ids'] == [owner for owner in v5_order if owner not in annotation_covered]
    assert summary['covered_owner_count'] + summary['missing_owner_count'] == summary['target_owner_count'] == len(v4_order)
    assert len(summary['annotation_covered_owner_ids']) + len(summary['annotation_missing_owner_ids']) == summary['annotation_target_owner_count'] == len(v5_order)
    assert summary['physical_repeat_row_count'] == sum(d['physical_status'] == 'repeat' for d in decisions)
    assert summary['confirmed_false_row_count'] == sum(d['physical_status'] == 'false' for d in decisions)
    assert summary['physical_unknown_row_count'] == sum(d['physical_status'] == 'unknown' for d in decisions)
    assert summary['invalid_output_row_count'] == sum(d['physical_status'] == 'invalid_output' for d in decisions)
    assert summary['class_wrong_row_count'] == sum(d['class'] == 'wrong' for d in decisions)
    assert summary['class_unknown_row_count'] == sum(d['class'] == 'unknown' for d in decisions)
    viewed = {group for note in notes if note['phase'] in {'shared', expected_phase} for group in note['covered_visual_group_ids']}
    unmatched_groups = {rendered[prediction_id]['visual_group_id'] for prediction_id in inherited['unmatched_valid_prediction_ids']}
    assert unmatched_groups <= viewed
    for note in notes:
        assert note['evidence_sha256'] == sha256(Path(note['evidence_path']))
    return {
        'artifact': str(path),
        'raw_rows': len(decisions),
        'fixed_v4_covered': summary['covered_owner_count'],
        'fixed_v4_target': summary['target_owner_count'],
        'annotation_v5_covered': len(summary['annotation_covered_owner_ids']),
        'annotation_v5_target': summary['annotation_target_owner_count'],
        'invalid_output_rows': summary['invalid_output_row_count'],
        'physical_unknown_rows': summary['physical_unknown_row_count'],
        'stop_reason': summary['raw_stop_reason'],
        'cap_debt': summary['cap_debt'],
        'status': 'validated_candidate',
    }


def validate_only():
    notes = []
    with (OUT / 'review-notes.jsonl').open() as handle:
        for line in handle:
            notes.append(json.loads(line))
    parent = validate_review(OUT / 'parent64-review.json', 'parent64', 64, notes)
    final = validate_review(OUT / 'final256-review.json', 'final256', 256, notes)
    parent_review = read(OUT / 'parent64-review.json')
    final_review = read(OUT / 'final256-review.json')
    parent_ids = parent_review['summary']['covered_owner_ids']
    final_ids = final_review['summary']['covered_owner_ids']
    assert final_review['summary']['retained_parent_owner_ids'] == [owner for owner in parent_ids if owner in set(final_ids)]
    assert final_review['summary']['lost_parent_owner_ids'] == [owner for owner in parent_ids if owner not in set(final_ids)]
    assert final_review['summary']['newly_covered_fixed_owner_ids'] == [owner for owner in final_ids if owner not in set(parent_ids)]
    assert final_review['summary']['annotation_missing_owner_ids'] == ['third-fit:new:25274:orange-skirt-person']
    assert not parent_review['new_owner_candidates'] and not final_review['new_owner_candidates']
    print(json.dumps({'schema': 'fourth_fit_paired_owner_review.validation.v1', 'status': 'validated_candidate', 'image_id': IMAGE_ID, 'parent64': parent, 'final256': final}, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    if not args.validate_only:
        notes = review_notes()
        with (OUT / 'review-notes.jsonl').open('w') as handle:
            for note in notes:
                handle.write(json.dumps(note, sort_keys=True) + '\n')
        parent = make_review('parent-step-00064', 64, 'parent64-review.json')
        make_review('fourth-step-00256', 256, 'final256-review.json', retained_parent=parent)
    validate_only()


if __name__ == '__main__':
    main()
