"""Consumer-facing invariants for the bounded dynamics packet and burden ledger."""
import pytest

from probes.parallel_owner_research.dynamics import (
    BOX_START, COORD, ROW_END, ROW_START, behavior, choose_boundaries,
    native_record, select_alternatives,
)


def row(bins, desc=42):
    return [ROW_START, desc, 151647, BOX_START] + [COORD + b for b in bins] + [ROW_END]


def text(bins, desc='person'):
    return '<|object_ref_start|>' + desc + '<|object_ref_end|><|box_start|>' + ''.join(
        f'<|coord_{b}|>' for b in bins) + '<|box_end|>'


def parsed(value):
    golden = dict(row_id='x', row_index=0, example_id='x', gt=[], image_width=1000,
                  image_height=1000, image_path='/unused.jpg')
    return native_record(value, golden, golden, 'im_end')


def test_alternative_families_remain_distinct_and_overlap_is_not_double_execution():
    logits = [-float(i) for i in range(1000)]
    logits[10], logits[20], logits[30] = 10., 9., 8.
    selections = select_alternatives(logits, COORD)
    assert len(selections) == 7
    assert [x['coordinate_bin'] for x in selections if 'high_probability' in x['families']] == [10, 20, 30]
    assert [x['coordinate_bin'] for x in selections if 'geometric_neighbor' in x['families']] == [1, 2, 3]
    selections = select_alternatives([-float(i) for i in range(1000)], COORD)
    assert len(selections) == 4
    assert all(x['families'] == ['high_probability', 'geometric_neighbor'] for x in selections[1:])


def test_alternative_geometry_ties_and_range_are_frozen():
    selections = select_alternatives([0.] * 1000, COORD + 500)
    near = [x['coordinate_bin'] for x in selections if 'geometric_neighbor' in x['families']]
    assert near == [498, 499, 501]
    assert len({x['token_id'] for x in selections}) == len(selections)
    with pytest.raises(ValueError, match='nonfinite'):
        select_alternatives([float('nan')] * 1000, COORD)


def test_control_is_latest_native_nonrepeat_not_an_outcome_selected_neighbor():
    a, b = row([0, 0, 100, 100]), row([200, 200, 300, 300])
    h = a + b
    boundaries = choose_boundaries(h + b + [151645], h, dict(image_width=1000, image_height=1000))
    assert boundaries[0]['history_ids'] == h
    assert boundaries[1]['history_ids'] == a
    assert boundaries[1]['target_row']['ids'] == b
    with pytest.raises(ValueError, match='no immediate strict repeat'):
        choose_boundaries(h + row([400, 400, 500, 500]), h, dict(image_width=1000, image_height=1000))


def test_mixed_row_never_becomes_autonomous_and_invalid_rows_do_not_disappear():
    h = text([0, 0, 100, 100])
    mixed = text([1, 0, 100, 100])
    invalid = text([800, 0, 100, 100])
    later = text([0, 0, 100, 100], 'chair')
    value = h + mixed + invalid + later + '<|im_end|>'
    result = behavior(value, parsed(value), history_text=h,
                      intervention_char_end=len(h) + mixed.index('<|coord_1|>') + len('<|coord_1|>'), stop='im_end')
    assert result['first_intervened_row']['mixed_intervened_row']
    assert result['first_intervened_row']['strict_repeat']
    assert result['first_autonomous_complete_row']['reason'] == 'geometry_invalid'
    assert result['geometry_invalid_rows'] == 1
    assert result['autonomous_valid_rows'] == 1
    assert result['autonomous_strict_repeats'] == 1  # class-blind, counted once despite two predecessors
    assert result['posthistory_strict_repeats'] == 2


def test_strict_threshold_is_not_inclusive_and_malformed_tail_is_visible():
    h = text([0, 0, 100, 100])
    mixed = text([5, 0, 100, 100])  # IoU exactly .95: not a strict repeat
    tail = '<|object_ref_start|>person<|object_ref_end|><|box_start|>'
    value = h + mixed + tail
    result = behavior(value, parsed(value), history_text=h,
                      intervention_char_end=len(h) + mixed.index('<|coord_5|>') + len('<|coord_5|>'), stop='length')
    assert not result['first_intervened_row']['strict_repeat']
    assert result['other_parser_drops'] == 1
    assert result['raw_starts_after_history'] == 2
    assert result['cap']


def test_native_projection_not_bin_space_owns_repeat_identity():
    # One-bin displacement can disappear after the authoritative native pixel rounding.
    from src.data.geometry import coord_bins_to_pixel_xyxy, iou_xyxy
    a, b = [0, 0, 10, 100], [1, 0, 10, 100]
    assert iou_xyxy(a, b) < .95
    ap = coord_bins_to_pixel_xyxy(a, image_width=100, image_height=100, field='a')
    bp = coord_bins_to_pixel_xyxy(b, image_width=100, image_height=100, field='b')
    assert iou_xyxy(ap, bp) == 1.


def test_real_consumer_excludes_entire_intervention_row_from_owner_credit():
    from types import SimpleNamespace
    from probes.parallel_owner_research.dynamics import parse_cell
    mapping = {ROW_START: '<|object_ref_start|>', 42: 'person', 151647: '<|object_ref_end|>',
               BOX_START: '<|box_start|>', ROW_END: '<|box_end|>', 151645: '<|im_end|>'}
    def decode(ids, skip_special_tokens=False):
        return ''.join(mapping[t] if t in mapping else f'<|coord_{t - COORD}|>' for t in ids)
    qwen = SimpleNamespace(tokenizer=SimpleNamespace(decode=decode))
    golden = dict(row_id='x', row_index=0, example_id='x', image_width=1000,
                  image_height=1000, image_path='/unused.jpg',
                  gt=[dict(bbox=[100, 100, 200, 200], description='person', object_id='owner1')])
    full = row([100, 100, 200, 200]) + [151645]
    case = dict(source_case=golden, golden=golden)
    result = parse_cell(qwen, case, dict(history_ids=[]), full[:5], full[5:], 'im_end')
    assert result['full_score_descriptive']['50']['tp'] == 1
    assert result['autonomous_score']['50']['tp'] == 0
    assert result['autonomous_score']['50']['owners'] == []
