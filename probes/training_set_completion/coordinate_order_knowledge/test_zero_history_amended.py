import copy
import json

import pytest

from probes.training_set_completion.coordinate_order_knowledge import distribution, probe, zero_history_amended


def test_actual_prefix_role_count_and_current_row_mutations_fail_closed():
    old = json.loads(zero_history_amended.OLD_DISTRIBUTION.read_text())
    knowledge = json.loads(zero_history_amended.OLD_KNOWLEDGE.read_text())
    case = next(x for x in knowledge['cases'] if x['image_id'] == 885)
    q = next(x['query'] for x in old['items'] if x['image_id'] == 885 and
             x['query']['relative_row'] == 4 and x['role'] == 'x2')
    tokens = [t for t in probe.jsonl(probe.SOURCE / 'pred_token_trace.jsonl')
              if t['row_id'] == case['row_id'] and t['trace_type'] == 'generated_token' and not t['is_pad']]
    tokens.sort(key=lambda t: t['generated_step_index'])
    rows = zero_history_amended.eligible_rows(probe.boxes(tokens), q['box_start'])
    assert len(rows) == 4
    prompt = case['prompt_token_ids']
    native = {'name': 'native', 'role': None, 'subset': None, 'replacement': None,
              'row_starts': [], 'indices': [], 'edited_prefix_sha256': q['prefix_sha256'],
              'full_history_sha256': probe.digest_json([*prompt, *q['prefix_token_ids']])}
    conditions = [native, *zero_history_amended._make_conditions(q, rows, prompt, 1),
                  *zero_history_amended._make_conditions(q, rows, prompt, 47)]
    zero_history_amended.validate_conditions(q, rows, prompt, conditions)
    for mutated in (
        ('wrong-role', lambda c: c[2].update(role='y2')),
        ('wrong-index', lambda c: c[2].update(indices=c[1]['indices'])),
        ('mismatched-count', lambda c: c[4].update(indices=c[4]['indices'][:-1])),
        ('current-row', lambda c: c[1].update(indices=[q['box_start'] + 2])),
        ('changed-current-prefix', lambda c: c[1].update(edited_prefix_sha256='0' * 64)),
    ):
        trial = copy.deepcopy(conditions)
        mutated[1](trial)
        with pytest.raises(ValueError):
            zero_history_amended.validate_conditions(q, rows, prompt, trial)


def test_four_frozen_target_row_counts_and_replacement_source():
    old = json.loads(zero_history_amended.OLD_DISTRIBUTION.read_text())
    stage = json.loads((distribution.ROOT / 'reduction/stage-a-v1.json').read_text())
    expected = {(885, 4): 4, (885, 8): 8, (5586, 2): 2, (5586, 3): 3}
    traces = zero_history_amended._source_trace()
    for image_id, relative in zero_history_amended.zero_history.TARGETS:
        q = next(x['query'] for x in old['items'] if x['image_id'] == image_id and
                 x['query']['relative_row'] == relative and x['role'] == 'x2')
        rows = zero_history_amended.eligible_rows(probe.boxes(traces[image_id]), q['box_start'])
        assert len(rows) == expected[(image_id, relative)]
        assert all(r['bins'][1] == r['bins'][2] == 0 and r['box_start'] < q['box_start'] for r in rows)
        first = next(r for r in stage['rows'] if r['image_id'] == image_id and
                     r['kind'] == 'invalid_native' and r['role'] == 'x2')
        assert first['metrics']['best_legal_bin'] == 47
