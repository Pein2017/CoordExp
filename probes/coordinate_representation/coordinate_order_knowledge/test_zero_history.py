import json

from probes.coordinate_representation.coordinate_order_knowledge import distribution, probe, zero_history


def test_same_row_match_requires_both_zero_before_current_row():
    boxes = [{'box_start': 0, 'coord_bins': [0, 0, 0, 38]},
             {'box_start': 9, 'coord_bins': [0, 0, 47, 0]},
             {'box_start': 18, 'coord_bins': [0, 0, 0, 0]}]
    before_current = zero_history.prior_rows(boxes, 18)
    assert before_current['x2_zero'] == before_current['y2_zero'] == 1
    assert before_current['same_row_x2_y2_zero'] == 0
    assert before_current['same_row_y1_x2_zero'] == 1
    assert zero_history.prior_rows(boxes, 19)['same_row_x2_y2_zero'] == 1


def test_frozen_target_prefixes_have_no_matched_historical_zero_rows():
    old = json.loads((distribution.ROOT / 'selection/plan-v2.json').read_text())
    row_ids = {probe.row_id(i) for i, _ in zero_history.TARGETS}
    trace = {rid: [] for rid in row_ids}
    for t in probe.jsonl(probe.SOURCE / 'pred_token_trace.jsonl'):
        if t['row_id'] in trace and t['trace_type'] == 'generated_token' and not t['is_pad']:
            trace[t['row_id']].append(t)
    expected_x2 = {(885, 4): 4, (885, 8): 8, (5586, 2): 2, (5586, 3): 3}
    for image_id, relative in zero_history.TARGETS:
        tokens = sorted(trace[probe.row_id(image_id)], key=lambda t: t['generated_step_index'])
        q = next(i['query'] for i in old['items'] if i['image_id'] == image_id and
                 i['query']['relative_row'] == relative and i['role'] == 'x2')
        ids = [int(t['token_id']) for t in tokens]
        assert q['prefix_sha256'] == probe.digest_json(ids[:q['query_index']])
        assert q['query_index'] == q['box_start'] + 3
        assert ids[q['box_start'] + 1] == probe.COORD0
        counts = zero_history.prior_rows(probe.boxes(tokens), q['box_start'])
        assert counts['x2_zero'] == expected_x2[(image_id, relative)]
        assert counts['y2_zero'] == counts['same_row_x2_y2_zero'] == 0
