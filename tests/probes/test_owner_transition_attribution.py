"""Frozen evaluator distinctions and serialized consumer falsifiers."""
import copy
import hashlib
import json

import pytest

from probes import owner_transition_attribution as x


SPECIAL = {'<|object_ref_start|>': 1, '<|object_ref_end|>': 2, '<|box_start|>': 3, '<|box_end|>': 4,
           **{f'<|coord_{i}|>': 100 + i for i in range(1000)}}


def fixture(objects, predictions):
    image = dict(image_id=7, cohort='fixture', objects=[dict(coco_ann_id=j, desc=d, bbox_2d=b) for j, (d, b) in enumerate(objects)])
    text, ids = '', []
    for desc, box in predictions:
        text += '<|object_ref_start|>' + desc + '<|object_ref_end|><|box_start|>' + ''.join(f'<|coord_{b}|>' for b in box) + '<|box_end|>'
        ids.extend([1, 2000, 2, 3, *[100+b for b in box], 4])
    raw = dict(image_id=7, request_id='7:greedy:0', arm='greedy', width=1000, height=1000,
               crop=[0, 0, 1000, 1000], text=text, token_ids=ids, generated_tokens=len(ids), stop_reason='im_end', raw_identity='fixture')
    measurement = x.rows.assess_outputs([image], [], [raw])[0]
    pool = x.positioned_pool(raw, SPECIAL)
    record = dict(path='fixture.json', raw=raw)
    return measurement, [x.side_witness(image, o, record, measurement, pool) for o in image['objects']]


def test_real_caller_class_agnostic_postassignment_and_all_duplicates():
    # Category-aware rematching could credit both; the maintained class-agnostic
    # IoU objective takes the two exact boxes and credits neither.
    m, sides = fixture([('person', [0, 0, 100, 100]), ('car', [20, 0, 120, 100])],
                       [('car', [0, 0, 100, 100]), ('person', [20, 0, 120, 100])])
    assert m['ids']['raw']['retained'] == [0, 1] and m['ids']['category']['retained'] == []
    assert m['burdens']['category_disagreements'] == 2
    assert all(x.classify(s) == x.CLASSES[0] and s['assignment_status'] == 'assigned_different_description' for s in sides)
    assert sides[0]['eligible_same_description'][0]['assigned_to']['reference_owner_id'] == '1'
    _, sides = fixture([('person', [0, 0, 100, 100])] * 2, [('person', [0, 0, 100, 100])])
    assert x.classify(sides[1]) == x.CLASSES[0] and sides[1]['assignment_status'] == 'unassigned'
    assert sides[1]['eligible_same_description'][0]['assigned_to']['reference_owner_id'] == '0'
    m, _ = fixture([('person', [0, 0, 100, 100])] * 2, [('person', [0, 0, 100, 100])] * 2)
    assert m['ids']['category']['retained'] == [0, 1] and m['burdens']['literal_valid_repeats'] == 1


def test_real_caller_cardinality_exact_half_and_parser_drop_gap():
    # Reuses tests/eval/test_assignment.py's greedy-versus-cardinality geometry.
    m, _ = fixture([('person', [0, 0, 100, 100]), ('person', [40, 0, 140, 100])],
                   [('person', [20, 0, 120, 100]), ('person', [0, 0, 60, 100])])
    assert [(z['reference_index'], z['prediction_order']) for z in m['matches']] == [(0, 1), (1, 0)]
    m, sides = fixture([('person', [0, 0, 100, 100])],
                       [('person', [0, 0, 0, 100]), ('person', [0, 0, 50, 100])])
    assert m['burdens']['geometry_invalid'] == 1 and m['matches'][0]['iou'] == .5
    witness = sides[0]['actual_assignment']
    assert witness['generated_order'] == 0 and witness['raw_parser_order'] == 1
    assert witness['prediction_id'].endswith(':p1') and witness['token_interval'] == [9, 18]
    assert witness['coordinate_positions'] == [13, 14, 15, 16]


def test_four_classes_null_and_zero_through_caller():
    _, sides = fixture([('person', [0, 0, 100, 100])], [('car', [0, 0, 100, 100])])
    assert x.classify(sides[0]) == x.CLASSES[1] and sides[0]['best_same_description'] is None
    _, sides = fixture([('person', [0, 0, 100, 100])], [('person', [0, 0, 49, 100])])
    assert x.classify(sides[0]) == x.CLASSES[2] and sides[0]['best_same_description']['iou_to_reference'] == .49
    _, absent = fixture([('person', [0, 0, 100, 100])], [])
    _, zero = fixture([('person', [0, 0, 100, 100])], [('person', [200, 0, 300, 100])])
    assert x.classify(absent[0]) == x.classify(zero[0]) == x.CLASSES[3]
    assert absent[0]['best_same_description'] is None and zero[0]['best_same_description']['iou_to_reference'] == 0


@pytest.fixture(scope='module')
def actual():
    inputs = x.bound_inputs()
    value = x.projection(inputs)
    value['analysis_source'] = x.analysis_source()
    return inputs, value


def test_actual_54_serialized_consumer_and_resigned_falsifiers(actual, tmp_path):
    inputs, value = actual
    path = tmp_path / 'attribution.json'
    path.write_text(x.canonical(value))
    original_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    assert x.consume(path, inputs)['instances'] == 114
    assert value['reconciliation']['transition_counts'] == {'R-single': {'gain': 31, 'loss': 29}, 'R-multiple': {'gain': 24, 'loss': 30}}
    def mutations(v, kind):
        e = v['entries'][0]
        covered = e['sides'][e['arm'] if e['direction'] == 'gain' else 'anchor']
        if kind == 'fabricated_id': e.update(annotation_id=123456789, instance_id='fabricated/instance')
        elif kind == 'omitted_id': v['entries'].pop()
        elif kind == 'assignment': covered['actual_assignment']['assigned_to']['reference_owner_id'] = '123456789'
        elif kind == 'iou': covered['actual_assignment']['iou_to_reference'] = .123
        elif kind == 'classification': e['classification'] = 'fabricated_class'
        elif kind == 'positions': covered['actual_assignment']['coordinate_positions'][0] += 1
        elif kind == 'compressed_vs_raw': covered['actual_assignment']['raw_parser_order'] += 1
        elif kind == 'source_diff': v['analysis_source']['diff'] += '\nfabricated source\n'
    for kind in ('fabricated_id', 'omitted_id', 'assignment', 'iou', 'classification', 'positions', 'compressed_vs_raw', 'source_diff'):
        changed = copy.deepcopy(value)
        mutations(changed, kind)
        path.write_text(x.canonical(changed))
        refreshed_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        assert refreshed_sha != original_sha
        # A freshly signed artifact remains invalid: the consumer reads raw inputs.
        message = 'analysis source revision/diff differs' if kind == 'source_diff' else 'attribution differs from bound raw records'
        with pytest.raises(ValueError, match=message):
            x.consume(path, inputs)


def test_inherited_measurement_mismatch_fails_closed(actual):
    inputs = copy.deepcopy(actual[0])
    inputs['records']['anchor'][0]['measurement']['burdens']['valid_rows'] += 1
    with pytest.raises(ValueError, match='inherited evaluator mismatch'):
        x.projection(inputs)
