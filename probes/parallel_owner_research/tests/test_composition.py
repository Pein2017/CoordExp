import pytest
import re
from types import SimpleNamespace

from probes.parallel_owner_research.composition import acquisition_job, complete_row_span, qualify_sample


def score(owners, **debt):
    return {'50': {'owners': owners, 'fp': debt.pop('fp', 0)},
            'strict_repeats': 0, 'parser_drops': 0,
            'invalid_predictions': 0, 'cap': 0, **debt}


def test_stable_anchor_not_source_gained_list_owns_pair_supply():
    q = qualify_sample(score(['1', '2', '3']), score(['1', '2']))
    assert q['stable_missed_owner_ids'] == ['3']
    assert not q['at_least_two_stable_missed']
    assert not q['physical_omission_admitted']


def test_invalid_burden_and_old_owner_loss_are_never_hidden():
    q = qualify_sample(score(['2', '3'], invalid_predictions=1), score(['1']))
    assert q['at_least_two_stable_missed']
    assert q['stable_owner_losses'] == ['1']
    assert not q['retains_all_stable_owners']
    assert not q['sample_burden_not_worse']


class Tokens:
    table = {'p': 7, 'a': 151646, 'b': 9, 'c': 151649}

    def decode(self, ids, **kwargs):
        reverse = {v: k for k, v in self.table.items()}
        return ''.join(reverse[v] for v in ids)

    def encode(self, text, **kwargs):
        class Encoded:
            ids = [self.table[c] for c in text]
        return Encoded()


def test_full_row_span_keeps_exact_prefix_and_all_row_tokens():
    row = complete_row_span([7, 151646, 9, 151649],
                           {'char_start': 1, 'char_end': 4, 'raw_span_text': 'abc'}, Tokens())
    assert row['prefix_token_ids'] == [7]
    assert row['target_token_ids'] == [151646, 9, 151649]


def test_complete_span_has_teeth_against_partial_or_multiple_rows():
    with pytest.raises(ValueError, match='complete object row'):
        complete_row_span([7, 151646, 9, 151649],
                          {'char_start': 1, 'char_end': 3, 'raw_span_text': 'ab'}, Tokens())
    with pytest.raises(ValueError, match='more than one row'):
        complete_row_span([151646, 9, 151649, 151646, 9, 151649],
                          {'char_start': 0, 'char_end': 6, 'raw_span_text': 'abcabc'}, Tokens())


def test_acquisition_roles_preserve_fixed_prefix_and_do_not_steal_free_budget():
    case = {'common_prefix_token_ids': [151646, 1, 151649],
            'targets': [{'owner_id': 'A', 'target_token_ids': [151646, 2, 151649]},
                        {'owner_id': 'B', 'target_token_ids': [151646, 3, 151649]}]}
    a = acquisition_job(case, 'A')
    b = acquisition_job(case, 'B')
    ab = acquisition_job(case, 'AB')
    assert a['prefix_ids'] == b['prefix_ids'] == ab['prefix_ids']
    assert ab['forced_ids'] == a['forced_ids'] + b['forced_ids']
    assert ab['forced_owners'] == ['A', 'B']
    assert ab['remaining_budget'] == 3084 - 9
    assert acquisition_job(case, 'natural')['remaining_budget'] == 3084
    case['common_prefix_token_ids'].append(151645)
    with pytest.raises(ValueError, match='nonterminal'):
        acquisition_job(case, 'A')


def test_residual_condition_is_P_plus_A_not_competing_target_at_P():
    from probes.parallel_owner_research.composition import residual_job
    case = {'common_prefix_token_ids': [151646, 1, 151649],
            'targets': [{'owner_id': 'A', 'target_token_ids': [151646, 2, 151649]},
                        {'owner_id': 'B', 'target_token_ids': [151646, 3, 151649]}]}
    job = residual_job(case, 'residual_B')
    assert job['prefix_ids'] == [151646, 1, 151649, 151646, 2, 151649]
    assert job['forced_ids'] == [151646, 3, 151649]
    assert job['forced_owners'] == ['B']
    assert job['remaining_budget'] == 3084 - 9
    with pytest.raises(ValueError, match='exact residual branch'):
        residual_job(case, 'natural')


def test_residual_admission_preserves_actual_A_path_not_just_Stable50_owner():
    from probes.parallel_owner_research.composition import residual_outcome
    baseline = score(['old', 'A', 'A_path_incumbent'])
    changed = score(['old', 'A', 'B'])
    result = residual_outcome(baseline, changed, new_owner='B', forced_assigned=True,
                              baseline_burden={'geometry_invalid': 0, 'other_malformed': 0},
                              current_burden={'geometry_invalid': 0, 'other_malformed': 0}, stop='im_end')
    assert not result['incrementally_useful']
    assert result['lost_A_path_owners'] == ['A_path_incumbent']
    assert 'A_path_owner_loss' in result['reasons']


@pytest.mark.parametrize('failure', ['force_not_assigned', 'invalid', 'malformed', 'cap', 'FP'])
def test_residual_target_gain_cannot_hide_wrong_forcing_or_output_debt(failure):
    from probes.parallel_owner_research.composition import residual_outcome
    baseline = score(['old', 'A'])
    changed = score(['old', 'A', 'B'], fp=int(failure == 'FP'))
    result = residual_outcome(baseline, changed, new_owner='B',
                              forced_assigned=failure != 'force_not_assigned',
                              baseline_burden={'geometry_invalid': 0, 'other_malformed': 0},
                              current_burden={'geometry_invalid': int(failure == 'invalid'),
                                              'other_malformed': int(failure == 'malformed')},
                              stop='length' if failure == 'cap' else 'im_end')
    assert not result['incrementally_useful']


def test_residual_success_requires_entire_A_set_plus_new_owner():
    from probes.parallel_owner_research.composition import residual_outcome
    result = residual_outcome(score(['old', 'A', 'incumbent']), score(['old', 'A', 'incumbent', 'B']),
                              new_owner='B', forced_assigned=True,
                              baseline_burden={'geometry_invalid': 0, 'other_malformed': 0},
                              current_burden={'geometry_invalid': 0, 'other_malformed': 0}, stop='im_end')
    assert result['incrementally_useful']
    assert result['gained_vs_A'] == ['B']


class ObjectTokens:
    """Deterministic token fixture; the REAL native parser/matcher is not mocked."""
    special = {'<|object_ref_start|>': 151646, '<|object_ref_end|>': 151647,
               '<|box_start|>': 151648, '<|box_end|>': 151649, '<|im_end|>': 151645,
               'person': 17}

    def encode(self, text, **kwargs):
        pieces = [p for p in re.split(r'(<\|[^>]+\|>)', text) if p]
        return SimpleNamespace(ids=[self.special[p] if p in self.special else 152000 + int(p[8:-2])
                                    for p in pieces])

    def decode(self, ids, **kwargs):
        reverse = {v: k for k, v in self.special.items()}
        return ''.join(reverse[i] if i in reverse else f'<|coord_{i - 152000}|>' for i in ids)


def residual_native_fixture(*, include_incumbent=True):
    from probes.dora_owner_learning.candidate_opportunity import score as native_score
    from probes.parallel_owner_research.composition import output_burden, residual_job
    from src.eval.native_rows import native_detection_record as native_record
    tok = ObjectTokens()
    boxes = {'old': [10, 10, 100, 100], 'A': [210, 210, 300, 300],
             'incumbent': [410, 410, 500, 500], 'B': [610, 610, 700, 700]}
    def row_text(owner):
        return '<|object_ref_start|>person<|object_ref_end|><|box_start|>' + ''.join(
            f'<|coord_{v}|>' for v in boxes[owner]) + '<|box_end|>'
    ids = {owner: tok.encode(row_text(owner)).ids for owner in boxes}
    golden = {'example_id': 'fixture', 'row_id': 'fixture', 'row_index': 0,
              'image_width': 1000, 'image_height': 1000, 'image_path': '/unused.png',
              'gt': [{'object_id': owner, 'description': 'person', 'bbox': box} for owner, box in boxes.items()]}
    base_ids = ids['old'] + ids['A'] + ids['incumbent'] + [151645]
    baseline = native_record(tok.decode(base_ids), {'row_id': 'fixture'}, golden, 'im_end')
    base_score = native_score(baseline, seed=None, length=len(base_ids), stop='im_end')
    case = {'image_id': 1, 'example_id': 'fixture', 'image': {'row_id': 'fixture'},
            'A': 'A', 'B': 'B', 'common_prefix_token_ids': ids['old'], 'stable_parsed': golden,
            'targets': [{'owner_id': owner, 'target_token_ids': ids[owner], 'target_text': row_text(owner)}
                        for owner in ('A', 'B')], 'backward_x_transition': False,
            'residual_baseline': {'score': base_score, 'burden': output_burden(baseline, base_ids, base_score)}}
    job = residual_job(case, 'residual_B')
    free = (ids['incumbent'] if include_incumbent else []) + [151645]
    action = job['prefix_ids'] + job['forced_ids'] + free
    text = tok.decode(action)
    raw = {'image_id': 1, 'example_id': 'fixture', 'stage': 'residual_B', **job,
           'free_ids': free, 'action_ids': action, 'text': text, 'stop_reason': 'im_end',
           'parsed': native_record(text, {'row_id': 'fixture'}, golden, 'im_end')}
    return [raw], {'records': [case]}, tok


def test_native_residual_consumer_accepts_real_row_binding_and_flags_incumbent_loss():
    from probes.parallel_owner_research.composition import reduce_residual_admission
    raw, packet, tok = residual_native_fixture()
    result = reduce_residual_admission(raw, packet, tok)['results'][0]
    assert result['incrementally_useful']
    assert result['forced_B_assigned']
    assert result['score']['50']['tp'] == 4
    raw, packet, tok = residual_native_fixture(include_incumbent=False)
    result = reduce_residual_admission(raw, packet, tok)['results'][0]
    assert not result['incrementally_useful']
    assert result['lost_A_path_owners'] == ['incumbent']


def test_native_residual_consumer_rejects_a_rebound_condition_and_duplicate_case():
    from probes.parallel_owner_research.composition import reduce_residual_admission
    raw, packet, tok = residual_native_fixture()
    with pytest.raises(ValueError, match='denominator'):
        reduce_residual_admission(raw + raw, packet, tok)
    raw[0]['prefix_ids'] = packet['records'][0]['common_prefix_token_ids']
    with pytest.raises(ValueError, match='exact P\\+A'):
        reduce_residual_admission(raw, packet, tok)
