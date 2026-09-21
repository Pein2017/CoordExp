import pytest

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
