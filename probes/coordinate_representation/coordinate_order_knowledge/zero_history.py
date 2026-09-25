"""CPU admission for the frozen same-row x2/y2-zero history contrast."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from probes.coordinate_representation.coordinate_order_knowledge import distribution, probe


ROOT = Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-23-coordinate-zero-history-feedback')
TARGETS = ((885, 4), (885, 8), (5586, 2), (5586, 3))


def prior_rows(boxes: list[dict], box_start: int) -> dict:
    """Literal complete rows strictly before the queried row, with role intersections."""
    earlier = [b for b in boxes if b['box_start'] < box_start]
    x2 = [b for b in earlier if b['coord_bins'][2] == 0]
    y2 = [b for b in earlier if b['coord_bins'][3] == 0]
    both = [b for b in x2 if b['coord_bins'][3] == 0]
    return {
        'complete_prior_rows': len(earlier),
        'x2_zero': len(x2),
        'y2_zero': len(y2),
        'same_row_x2_y2_zero': len(both),
        'same_row_x1_x2_zero': sum(b['coord_bins'][0] == 0 for b in x2),
        'same_row_y1_x2_zero': sum(b['coord_bins'][1] == 0 for b in x2),
        'x2_zero_rows': [{'box_start': b['box_start'], 'bins': b['coord_bins']} for b in x2],
        'paired_rows': [{'box_start': b['box_start'], 'bins': b['coord_bins']} for b in both],
    }


def prepare() -> Path:
    old = json.loads((distribution.ROOT / 'selection/plan-v2.json').read_text())
    stage_a = json.loads((distribution.ROOT / 'reduction/stage-a-v1.json').read_text())
    by_id = {c['image_id']: c for c in old['cases']}
    traces = defaultdict(list)
    selected = set(by_id)
    for token in probe.jsonl(probe.SOURCE / 'pred_token_trace.jsonl'):
        if token['trace_type'] == 'generated_token' and not token['is_pad'] and int(token['row_id'][-12:]) in selected:
            traces[int(token['row_id'][-12:])].append(token)
    for tokens in traces.values():
        tokens.sort(key=lambda t: t['generated_step_index'])
        if [t['generated_step_index'] for t in tokens] != list(range(len(tokens))):
            raise ValueError('noncontiguous accepted trace')
    targets = []
    for image_id, relative in TARGETS:
        case = by_id[image_id]
        item = next(x for x in old['items'] if x['image_id'] == image_id and
                    x['query']['relative_row'] == relative and x['role'] == 'x2')
        q = item['query']
        ids = [int(t['token_id']) for t in traces[image_id]]
        if (q['prefix_sha256'] != probe.digest_json(ids[:q['query_index']]) or
                ids[q['query_index']] != q['observed_token_id'] or
                q['box_start'] + 3 != q['query_index'] or
                ids[q['box_start'] + 1] != probe.COORD0 or
                q['original_threshold'] != 0):
            raise ValueError('frozen source prefix, causal x2 index or current x1 changed')
        first = next(x for x in stage_a['rows'] if x['image_id'] == image_id and
                     x['kind'] == 'invalid_native' and x['role'] == 'x2')
        replacement = first['metrics']['best_legal_bin']
        if replacement in (None, 1):
            raise ValueError('frozen legal replacement is unavailable or equals one')
        cell = distribution.ROOT / 'cells' / f"{image_id}-x2-{q['query_index']}.json"
        distribution._read_item(item)
        if not cell.is_file():
            raise FileNotFoundError(cell)
        targets.append({'image_id': image_id, 'relative_row': relative,
                        'query_index': q['query_index'], 'box_start': q['box_start'],
                        'prefix_sha256': q['prefix_sha256'],
                        'native_cell': probe.binding(cell),
                        'current_x1': q['original_threshold'], 'native_emitted_bin': q['observed_successor'],
                        'replacement_values_if_eligible': [1, replacement],
                        'history': prior_rows(probe.boxes(traces[image_id]), q['box_start'])})
    controls = []
    for case in old['cases']:
        if case['status'] != 'healthy_control':
            continue
        boxes = probe.boxes(traces[case['image_id']])
        eligible = [{'query_index': b['box_start'] + 3,
                     'paired_rows': prior_rows(boxes, b['box_start'])['paired_rows']}
                    for b in boxes if prior_rows(boxes, b['box_start'])['same_row_x2_y2_zero']]
        controls.append({'image_id': case['image_id'], 'complete_rows': len(boxes),
                         'x2_zero_rows': sum(b['coord_bins'][2] == 0 for b in boxes),
                         'y2_zero_rows': sum(b['coord_bins'][3] == 0 for b in boxes),
                         'same_row_x2_y2_zero_rows': sum(b['coord_bins'][2:] == [0, 0] for b in boxes),
                         'eligible_x2_queries': eligible})
    if any(x['history']['same_row_x2_y2_zero'] for x in targets) or any(x['eligible_x2_queries'] for x in controls):
        raise ValueError('CPU HOLD premise changed; freeze a full intervention packet instead')
    path = ROOT / 'selection/admission-v1.json'
    probe.write_json(path, {
        'schema': 'coordinate_zero_history_feedback.admission.v1', 'status': 'HOLD',
        'reason': 'no prior complete same-row x2=0 AND y2=0 pair at any target; no eligible healthy control',
        'source': {'old_distribution_plan': probe.binding(distribution.ROOT / 'selection/plan-v2.json'),
                   'old_distribution_acceptance': probe.binding(distribution.ROOT / 'lead-acceptance-v1.json'),
                   'old_stage_a': probe.binding(distribution.ROOT / 'reduction/stage-a-v1.json'),
                   'generated_trace': probe.binding(probe.SOURCE / 'pred_token_trace.jsonl'),
                   'source_config': probe.binding(probe.SOURCE / 'configs/resolved.json'),
                   'current_probe': probe.binding(Path(probe.__file__)),
                   'current_distribution': probe.binding(Path(distribution.__file__)),
                   'current_admission': probe.binding(Path(__file__))},
        'targets': targets, 'healthy_controls': controls,
        'counts': {'frozen_targets': len(targets), 'eligible_targets': 0,
                   'frozen_healthy_traces': len(controls), 'eligible_healthy_queries': 0,
                   'model_forwards': 0, 'allocated_gpu_seconds': 0},
        'minimal_alternative_for_lead': 'In the exact same historical x2-zero rows, y1 is also zero. Replacing historical y1-zero on those rows could preserve count, rows, zero-token status and a wrong-axis control; it changes the frozen y2 comparator and therefore requires a new lead contract. Frozen healthy traces still lack historical x2-zero rows.',
    })
    return path


if __name__ == '__main__':
    print(prepare())
