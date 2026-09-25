import json
import math
from pathlib import Path

import pytest
import torch

from probes.coordinate_representation.coordinate_order_knowledge.probe import COORD0, ROOT, boxes, case_queries, digest_json, readout, textual_pairs, thresholds, validate_query


def test_causal_role_and_boundary_falsification():
    ids = [151646, 8987, 151647, 151648, COORD0 + 600, COORD0 + 100,
           COORD0 + 601, COORD0 + 101, 151649]
    tokens = [{'token_id': x, 'token_text': str(x)} for x in ids]
    box = boxes(tokens)[0]
    x, y = case_queries(tokens, box, 'native')
    assert (x['query_index'], x['preceding_index'], x['wrong_axis_index']) == (6, 4, 5)
    assert (y['query_index'], y['preceding_index'], y['wrong_axis_index']) == (7, 5, 4)
    assert x['prefix_token_ids'] == ids[:6] and y['prefix_token_ids'] == ids[:7]
    assert {z['threshold'] for z in thresholds(0, 1)} == {0, 1, 4}
    assert {z['threshold'] for z in thresholds(998, 999)} == {994, 997, 998}
    wrong = [dict(t) for t in tokens]
    wrong[6]['token_id'] = COORD0 + 1001
    assert boxes(wrong) == []
    wrong = [dict(t) for t in tokens]
    wrong[5]['token_id'] = 42
    assert boxes(wrong) == []


def test_frozen_logit_null_is_not_a_redistribution():
    logits = torch.zeros(1200)
    ids = list(range(100, 1100))
    baseline = readout(logits, ids, 600)
    changed_threshold = readout(logits, ids, 604)
    assert changed_threshold['illegal_family_mass'] > baseline['illegal_family_mass']
    assert changed_threshold['coordinate_logprobs_conditional'] == baseline['coordinate_logprobs_conditional']
    logits[ids[600]] += 3
    redistributed = readout(logits, ids, 604)
    assert redistributed['illegal_family_mass'] > changed_threshold['illegal_family_mass']
    assert len(textual_pairs()) == 64


def test_frozen_plan_prefixes_and_roles():
    path = Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-23-coordinate-order-knowledge/selection/plan-v1.json')
    if not path.is_file():
        pytest.skip('selection not frozen in this checkout')
    plan = json.loads(path.read_text())
    assert len(plan['cases']) == 12 and len(plan['textual_pairs']) == 64
    for case in plan['cases']:
        for q in [*case['queries'], *(t for t in case['teacher'] if t.get('kind') == 'teacher')]:
            validate_query(q)
    q = dict(plan['cases'][0]['queries'][0])
    with pytest.raises(ValueError, match='shifted predecessor'):
        validate_query({**q, 'preceding_index': q['preceding_index'] + 1})
    with pytest.raises(ValueError, match='future suffix'):
        validate_query({**q, 'query_index': q['query_index'] + 1})
    with pytest.raises(ValueError, match='y2 history|swapped role'):
        validate_query({**q, 'role': 'y2'})


def test_saved_caller_conditions_and_frozen_logit_null():
    plan_path = ROOT / 'selection/plan-v1.json'
    terminal = ROOT / 'execution/terminal-v1.json'
    if not terminal.is_file():
        pytest.skip('no completed producer')
    plan = json.loads(plan_path.read_text())
    observed = 0
    for case in plan['cases']:
        for q in [*case['queries'], *(x for x in case['teacher'] if x.get('kind') == 'teacher')]:
            path = ROOT / 'cells' / f'{case["image_id"]}-{q["kind"]}-{q["role"]}-{q["query_index"]}.json'
            cell = json.loads(path.read_text())
            expected = {('same', x['threshold']) for x in q['thresholds']}
            expected |= {('wrong', x['threshold']) for x in q['thresholds'] if x['threshold'] != q['original_threshold']}
            assert {(x['axis'], x['requested_threshold']) for x in cell['conditions']} == expected
            assert len(cell['conditions']) == len(expected)
            baseline = next(x for x in cell['conditions'] if x['axis'] == 'same' and x['requested_threshold'] == q['original_threshold'])
            probabilities = [math.exp(x) for x in baseline['readout']['coordinate_logprobs_conditional']]
            assert abs(sum(probabilities) - 1) < 2e-5
            for condition in cell['conditions']:
                history = [*case['prompt_token_ids'], *q['prefix_token_ids']]
                if condition['axis'] == 'same':
                    history[len(case['prompt_token_ids']) + q['preceding_index']] = COORD0 + condition['requested_threshold']
                else:
                    history[len(case['prompt_token_ids']) + q['wrong_axis_index']] = COORD0 + condition['actual_changed_bin']
                assert condition['history_sha256'] == digest_json(history)
                null = sum(probabilities[:condition['observed_threshold'] + 1])
                assert abs(null - condition['frozen_logit_rethreshold_illegal_mass']) < 2e-6
                assert abs(condition['readout']['illegal_family_mass'] - null - condition['redistributed_illegal_mass']) < 2e-6
            observed += 1
    assert observed == 70
