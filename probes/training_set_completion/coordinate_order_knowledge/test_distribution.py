import copy
import json

import pytest
import torch

from probes.training_set_completion.coordinate_order_knowledge import distribution, probe


def _read(logits):
    z = torch.tensor(logits, dtype=torch.float32)
    return probe.readout(z, list(range(1000)), 600)


def test_distribution_distinguishes_diffuse_legal_from_sharp_modes_and_arithmetic_partition():
    diffuse = [-2.] * 1000 + [-10.]
    diffuse[600] = 5.
    a = distribution.metrics(_read(diffuse), 600, 600, fixed_threshold=600)
    assert a['illegal_minus_legal_logprob_margin'] > 0
    assert a['legal_entropy_normalized'] > .99
    assert a['legal_top_mass']['5'] < .02
    moved = distribution.metrics(_read(diffuse), 601, 600, fixed_threshold=600)
    assert moved['fixed_illegal_mass_conditional_coordinate'] == pytest.approx(a['fixed_illegal_mass_conditional_coordinate'])
    assert moved['illegal_mass_conditional_coordinate'] > moved['fixed_illegal_mass_conditional_coordinate']

    narrow = [-10.] * 1000 + [-10.]
    narrow[600], narrow[610], narrow[611], narrow[612], narrow[750] = 5., 4., 4., 4., 3.
    b = distribution.metrics(_read(narrow), 600, 600)
    assert b['illegal_minus_legal_logprob_margin'] > 0
    assert b['legal_entropy_normalized'] < a['legal_entropy_normalized']
    assert b['legal_top_mass']['5'] > .99
    assert b['legal_mass_near_top_bin']['1'] > .5
    assert b['legal_mass_near_top_bin']['1'] < b['legal_mass_near_top_bin']['5']
    assert distribution.metrics(_read(narrow), 998, 999)['legal_entropy_nats'] is None
    assert distribution.metrics(_read(narrow), 999, 999)['legal_entropy_nats'] is None


def test_real_trace_window_and_causal_predecessor_falsification():
    old = json.loads(distribution.OLD_PLAN.read_text())
    case = next(c for c in old['cases'] if c['image_id'] == 632)
    tokens = [x for x in probe.jsonl(probe.SOURCE / 'pred_token_trace.jsonl')
              if x['row_id'] == case['row_id'] and x['trace_type'] == 'generated_token' and not x['is_pad']]
    tokens.sort(key=lambda x: x['generated_step_index'])
    drops = next(x['dropped_predictions'] for x in probe.jsonl(probe.SOURCE / 'parse_diagnostics.jsonl')
                 if x['row_id'] == case['row_id'])
    rows, _ = distribution._window_queries(case, tokens, 'y2', drops)
    boundary = next(x for x in rows if x['relative_row'] == 0)
    assert boundary['query_index'] == 346 and boundary['original_threshold'] == 476
    assert boundary['observed_successor'] == 473 and boundary['original_role_illegal']
    broken = copy.deepcopy(boundary)
    broken['history_token_ids'] = [int(x['token_id']) for x in tokens]
    broken['original_threshold'] = 475
    with pytest.raises(ValueError, match='same-row causal threshold'):
        probe.validate_query(broken)
    shifted = copy.deepcopy(boundary)
    shifted['history_token_ids'] = [int(x['token_id']) for x in tokens]
    shifted['query_index'] += 1
    with pytest.raises(ValueError):
        probe.validate_query(shifted)
