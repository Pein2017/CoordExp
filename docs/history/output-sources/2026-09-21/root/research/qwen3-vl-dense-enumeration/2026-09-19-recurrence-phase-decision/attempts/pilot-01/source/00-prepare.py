"""Freeze chronological recurrence phases and the bounded native release matrix."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from probes.training_set_completion.numerical_feedback.select import rows, same

PANEL = Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-distribution-census/shared-panel.json')
ROOT = PANEL.parent.parent / '2026-09-19-recurrence-phase-decision'
BASE = 151670
LEGACY = (417044, 885, 5586, 7511, 14038, 632)


def binding(path: Path) -> dict:
    data = path.read_bytes()
    return {'path': str(path.resolve()), 'sha256': hashlib.sha256(data).hexdigest(), 'size_bytes': len(data)}


def write_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as file:
        json.dump(value, file, indent=2, allow_nan=False)
        file.write('\n')


def earliest_triple(rr: list[dict]) -> int | None:
    for i in range(len(rr) - 2):
        if all(same(rr[a], rr[b], 8) for a, b in ((i, i + 1), (i, i + 2), (i + 1, i + 2))):
            return i
    return None


def stage(row: dict, tokens: list[int]) -> dict:
    offsets = row['coordinate_offsets']
    desc_end = offsets[0] - 2
    result = {'row_index': row['index'], 'opener': row['start'],
              'description': list(range(row['start'] + 1, desc_end)),
              'description_end': desc_end, 'bbox_start': desc_end + 1,
              'coordinates': dict(zip(('x1', 'y1', 'x2', 'y2'), offsets, strict=True)),
              'bbox_end': row['end'] - 1,
              'next_boundary': row['end'] if row['end'] < len(tokens) else None}
    for key in ('opener', 'description_end', 'bbox_start', 'bbox_end'):
        assert tokens[result[key]] == {'opener': 151646, 'description_end': 151647,
                                        'bbox_start': 151648, 'bbox_end': 151649}[key]
    assert all(tokens[offset] == BASE + value for offset, value in zip(offsets, row['values'], strict=True))
    return result


def summarize(boundary: dict) -> dict:
    tokens = boundary['native_tokens']
    rr = rows(tokens)
    old = boundary['source_row']
    assert rr[old['index']] == old
    first = earliest_triple(rr)
    matches = [{'earlier': i, 'later': j, 'literal': same(rr[i], rr[j], 0)}
               for j in range(len(rr)) for i in range(j) if same(rr[i], rr[j], 8)]
    result = {'boundary_id': boundary['id'], 'model': boundary['model'],
              'split': boundary['split'], 'image_id': boundary['image_id'],
              'group': boundary['group'], 'batch_index': boundary['batch_index'],
              'kind': boundary['kind'], 'source_row_index': old['index'],
              'source_row_start': old['start'], 'source_row_end': old['end'],
              'earliest_triple_seed': first, 'earliest_triple_kind':
              ('literal' if first is not None and all(same(rr[a], rr[b], 0) for a, b in
               ((first, first + 1), (first, first + 2), (first + 1, first + 2))) else 'near') if first is not None else None,
              'earliest_numerical_match_later_row': matches[0]['later'] if matches else None,
              'matches_before_seed': [m for m in matches if m['later'] <= first] if first is not None else [],
              'rows_count': len(rr), 'invalid_row_indices': [r['index'] for r in rr if not r['valid']],
              'source': {key: boundary['bindings'][key] for key in ('raw', 'trace', 'receipt')},
              'source_panel': binding(Path(boundary['raw_path']).resolve().parents[3] / 'panel.json')}
    profile_indices = ([first - 1, first, first + 1, first + 2, first + 3]
                       if boundary['kind'] == 'failure' else [old['index'], old['index'] + 1])
    result['profile_rows'] = [dict(row_index=i, status='available', row=rr[i], stage=stage(rr[i], tokens))
                              if 0 <= i < len(rr) else dict(row_index=i, status='missing')
                              for i in profile_indices]
    result['phases'] = []
    for index in ([first, first + 1, first + 2] if boundary['kind'] == 'failure' else [old['index']]):
        row = rr[index]
        bridge = rr[index + 2] if index + 2 < len(rr) else None
        result['phases'].append({'name': ('seed', 'first_repeat', 'third_row')[index - first]
                                 if boundary['kind'] == 'failure' else 'proxy',
                                 'row_index': index, 'row': row, 'edit_offsets':
                                 dict(zip(('x1', 'y1', 'x2', 'y2'), row['coordinate_offsets'], strict=True)),
                                 'immediate_prefix_end': row['end'],
                                 'bridge_prefix_end': bridge['end'] if bridge else None,
                                 'bridge_row_indices': [index + 1, index + 2] if bridge else None,
                                 'hold': None if bridge else 'missing_two_complete_native_bridge_rows'})
    return result


def selection_key(item: dict) -> tuple:
    image = int(item['image_id'])
    try:
        return (0, LEGACY.index(image), '', 0, '')
    except ValueError:
        return (1, 0, item['split'], image, item['boundary_id'])


def make_plan(panel: dict) -> dict:
    summaries = [summarize(b) for b in panel['all_boundaries']]
    assert len(summaries) == 45
    by_model = ('tied', 'untied')
    selected = []
    for model in by_model:
        failures = sorted((s for s in summaries if s['model'] == model and s['kind'] == 'failure'), key=selection_key)
        proxies = sorted((s for s in summaries if s['model'] == model and s['kind'] != 'failure'),
                         key=lambda s: (s['image_id'] != 309264, s['split'], int(s['image_id']), s['boundary_id']))
        selected += failures[:8] + proxies[:2]
    assert len(selected) == 20 and sum(s['kind'] == 'failure' for s in selected) == 16
    cells, missing = [], []
    for source in selected:
        for phase in source['phases']:
            row = phase['row']
            for role in ('x1', 'y2'):
                k = {'x1': 0, 'y2': 3}[role]
                value = row['values'][k]
                for delta in (0, -1, 1):
                    if delta == 0 and role == 'y2':
                        continue  # one shared native control per phase and release boundary
                    if not 0 <= value + delta < 1000:
                        missing.append({'boundary_id': source['boundary_id'], 'phase': phase['name'], 'role': role,
                                        'delta': delta, 'reason': 'out_of_vocabulary'})
                        continue
                    changed = row['values'].copy()
                    changed[k] += delta
                    site = phase['edit_offsets'][role]
                    assert panel['all_boundaries'][next(i for i,b in enumerate(panel['all_boundaries']) if b['id']==source['boundary_id'])]['native_tokens'][site] == BASE + value
                    for mode, end in (('immediate', phase['immediate_prefix_end']), ('bridge', phase['bridge_prefix_end'])):
                        cell = {'id': f"{source['boundary_id']}__{phase['name']}__{role}{delta:+d}__{mode}",
                                'boundary_id': source['boundary_id'], 'model': source['model'], 'kind': source['kind'],
                                'phase': phase['name'], 'phase_row_index': phase['row_index'], 'role': role if delta else 'native',
                                'delta': delta, 'site_offset': site, 'old_token_id': BASE + value,
                                'new_token_id': BASE + value + delta, 'release_mode': mode,
                                'prefix_end': end, 'bridge_row_indices': phase['bridge_row_indices'] if mode == 'bridge' else [],
                                'geometry_valid': changed[0] < changed[2] and changed[1] < changed[3],
                                'native_geometry_valid': row['valid'], 'edited_values': changed,
                                'token_distance': end - site if end is not None and delta else None,
                                'status': 'ready' if end is not None else phase['hold']}
                        cells.append(cell)
    assert len(cells) + 2 * len(missing) <= 520 and len({c['id'] for c in cells}) == len(cells)
    return {'schema': 'recurrence_phase_decision.plan.v1', 'status': 'frozen',
            'panel': binding(PANEL), 'sources': binding(PANEL.with_name('shared-sources.json')),
            'source_summaries': summaries, 'selected_boundary_ids': [s['boundary_id'] for s in selected],
            'cells': cells, 'missing_directions': missing,
            'counts': {'panel_sources': len(summaries), 'selected_failure': 16, 'selected_proxy': 4,
                       'planned_cells': len(cells), 'ready_cells': sum(c['status'] == 'ready' for c in cells),
                       'held_cells': sum(c['status'] != 'ready' for c in cells), 'missing_directions': len(missing)}}


def selfcheck() -> None:
    def row(v): return [151646, 9, 151647, 151648, *[BASE + x for x in v], 151649]
    rr = rows(row([1, 2, 3, 4]) + row([2, 2, 3, 4]) + row([3, 2, 3, 4]) + row([10, 2, 20, 4]) * 3)
    assert earliest_triple(rr) == 0 and not same(rr[0], rr[2], 0)
    rr = rows(row([0, 1, 2, 3]) + row([7, 1, 2, 3]) + row([14, 1, 2, 3]))
    assert earliest_triple(rr) is None
    rr = rows(row([5, 2, 5, 4]) * 3)
    assert len(rr) == 3 and not rr[0]['valid'] and earliest_triple(rr) == 0
    assert stage(rr[0], row([5, 2, 5, 4]))['coordinates']['x1'] == 4
    print('PASS chronological pairwise triple, invalid retention, role offset')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--selfcheck', action='store_true')
    args = parser.parse_args()
    if args.selfcheck:
        selfcheck()
        return
    assert binding(PANEL)['sha256'] == '005bc6deff209bc96ce469e8cbc20d3dbd30a7e424a7ff18a0f7f8942a1bdb49'
    plan = make_plan(json.loads(PANEL.read_text()))
    write_new(ROOT / 'selection' / 'plan.json', plan)
    print(json.dumps(plan['counts']))


if __name__ == '__main__':
    main()
