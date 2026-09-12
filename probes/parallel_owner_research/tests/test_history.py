import copy

import pytest

from probes.parallel_owner_research import history


def row(value):
    return [151646, value, 151647, 151648, 151670, 151671, 151672, 151673, 151649]


def test_literal_earlier_swap_preserves_complete_multiset_and_final():
    p = row(4) + row(5) + row(6)
    q, permutation = history.earlier_swap(p)
    assert q == row(5) + row(4) + row(6)
    assert permutation == [1, 0, 2]
    history.validate_history_pair(p, q)


@pytest.mark.parametrize('p', [row(4) + row(5), row(4) + row(4) + row(6)])
def test_nontrivial_same_final_order_contrast_fails_closed(p):
    with pytest.raises(ValueError):
        history.earlier_swap(p)


@pytest.mark.parametrize('p', [row(4)[:-1], row(4) + [151645], [1] + row(4), row(4)[:4] + row(5)])
def test_prefix_never_silently_drops_incomplete_or_nonrow_tokens(p):
    with pytest.raises(ValueError):
        history.complete_rows(p)


def records():
    p = row(4) + row(5) + row(6)
    q, _ = history.earlier_swap(p)
    return [{'record_id': f'{cid}.{label}', 'case_id': cid,
             'prefix_token_ids': prefix, 'target_token_ids': row(9)}
            for cid in history.SELECTED for label, prefix in [('P', p), ('Q', q)]]


def test_equal_exposure_is_complete_row_not_equal_next_owner():
    ledger = history.validate_exposure(history.exposure_arms(), records())
    assert ledger['fixed_P']['record_exposures']['417044-c01.P'] == 32
    assert ledger['mixed_PQ']['record_exposures']['417044-c01.P'] == 16
    assert ledger['mixed_PQ']['record_exposures']['417044-c01.Q'] == 16
    assert ledger['fixed_P']['case_target_token_exposures'] == ledger['mixed_PQ']['case_target_token_exposures']


def test_changed_coordinate_target_is_not_prefix_only_even_if_owner_is_same():
    r = records()
    r[1]['target_token_ids'] = copy.deepcopy(r[1]['target_token_ids'])
    r[1]['target_token_ids'][4] += 1
    with pytest.raises(ValueError, match='literal complete target row changed'):
        history.validate_exposure(history.exposure_arms(), r)


def test_exposure_drift_has_teeth():
    arms = history.exposure_arms()
    arms['mixed_PQ']['steps'][0][0]['record_id'] = '417044-c01.Q'
    with pytest.raises(ValueError, match='exposure schedule'):
        history.validate_exposure(arms, records())


def test_last_row_and_coordinate_changes_rejected():
    p = row(4) + row(5) + row(6)
    with pytest.raises(ValueError, match='final row'):
        history.validate_history_pair(p, row(5) + row(6) + row(4))
    q, _ = history.earlier_swap(p)
    q[4] += 1
    with pytest.raises(ValueError, match='multiset'):
        history.validate_history_pair(p, q)


def text_row(coords, category='person'):
    return '<|object_ref_start|>' + category + '<|object_ref_end|><|box_start|>' + ''.join(f'<|coord_{n}|>' for n in coords) + '<|box_end|>'


def frozen():
    return {'case': {'row_id': 'fixture'}, 'golden': {'row_id': 'fixture', 'row_index': 0,
            'example_id': 'fixture', 'image_width': 999, 'image_height': 999, 'image_path': '/fixture.jpg',
            'gt': [{'object_id': 'owner', 'description': 'person', 'bbox': [100, 100, 200, 200]}]}}


def test_forced_target_gets_no_free_recovery_credit():
    target = text_row([100, 100, 200, 200])
    result = history.continuation_ledger(target, '<|im_end|>', frozen(), 9, 1, 'im_end')
    assert result['full_score']['50']['owners'] == ['owner']
    assert result['free_score']['50']['owners'] == []
    assert result['burden']['free_valid_rows'] == 0


def test_free_repeat_denominator_includes_supplied_history_once_and_keeps_invalid():
    target = text_row([100, 100, 200, 200])
    invalid = text_row([500, 500, 400, 400])
    result = history.continuation_ledger(target + target, target + invalid + '<|im_end|>', frozen(), 18, 19, 'im_end')
    assert result['burden']['free_strict_repeats_including_history'] == 1
    assert result['burden']['free_geometry_invalid'] == 1
    assert result['burden']['free_row_starts'] == 2
    assert result['burden']['free_valid_rows'] == 1
