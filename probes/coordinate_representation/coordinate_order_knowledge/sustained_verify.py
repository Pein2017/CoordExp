"""CPU-only replay of the frozen sustained-legality cells from saved tokens."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from probes.coordinate_representation.coordinate_order_knowledge import probe
from probes.coordinate_representation.coordinate_order_knowledge import sustained_legality as study


def _advance(phase: str, token: int, bins: dict[int, int], x1: int | None, y1: int | None,
             coords: list[int]) -> tuple[str, int | None, int | None, list[int]]:
    if phase == 'outside' and token == probe.REF_START:
        return 'description', None, None, []
    if phase == 'outside' and token == study.release.EOS:
        return 'stopped', None, None, []
    if phase == 'description' and token == probe.REF_END:
        return 'box_start', x1, y1, coords
    if phase == 'description' and token not in bins and token not in study.STRUCTURE:
        return phase, x1, y1, coords
    if phase == 'box_start' and token == probe.BOX_START:
        return 'x1', x1, y1, coords
    if phase in ('x1', 'y1', 'x2', 'y2') and token in bins:
        value = bins[token]
        coords = [*coords, value]
        if phase == 'x1':
            return 'y1', value, y1, coords
        if phase == 'y1':
            return 'x2', x1, value, coords
        if phase == 'x2':
            return 'y2', x1, y1, coords
        return 'box_end', x1, y1, coords
    if phase == 'box_end' and token == probe.BOX_END:
        return 'outside', None, None, coords
    raise ValueError(f'malformed saved token in {phase}')


def verify_cell(case: dict, cell: dict, coord_ids: list[int]) -> dict:
    if cell['case'] != case['image_id'] or cell['prefix_sha256'] != case['prefix_sha256'] or cell['status'] != 'complete':
        raise ValueError('cell identity/status differs from frozen case')
    prefix, emitted, steps = case['prefix_token_ids'], cell['token_ids'], cell['steps']
    if len(emitted) != len(steps) or len(emitted) > study.HORIZON:
        raise ValueError('token/step denominator changed')
    bins = {token: i for i, token in enumerate(coord_ids)}
    phase, x1, y1, coords = 'outside', None, None, []
    row_start = None
    complete = []
    for j, token in enumerate([*prefix, *emitted]):
        if j >= len(prefix):
            i = j - len(prefix)
            step = steps[i]
            legal = ([0, 998] if phase in ('x1', 'y1') else
                     [x1 + 1, 999] if phase == 'x2' and x1 is not None and x1 < 999 else
                     [y1 + 1, 999] if phase == 'y2' and y1 is not None and y1 < 999 else None)
            if phase in ('x1', 'y1', 'x2', 'y2') and legal is None:
                raise ValueError('empty legal coordinate family')
            if (step['index'] != i or step['emitted_token'] != token or step['stage'] != phase or
                    step['x1'] != x1 or step['y1'] != y1 or step['legal_range'] != legal or
                    step['history_sha256'] != probe.digest_json([*prefix, *emitted[:i]]) or
                    step['policy_argmax'] != token):
                raise ValueError(f'saved step state/mask/choice mismatch at {i}')
            if legal is None:
                if step['raw_argmax'] != token or abs(step['raw_logprob'] - step['policy_logprob']) > 2e-4:
                    raise ValueError('outside-coordinate token changed by policy')
            elif token not in coord_ids[legal[0]:legal[1] + 1]:
                raise ValueError('illegal emitted coordinate at slot')
        if phase == 'outside' and token == probe.REF_START:
            row_start = j
        old_phase = phase
        phase, x1, y1, coords = _advance(phase, token, bins, x1, y1, coords)
        if old_phase == 'box_end':
            if len(coords) != 4 or row_start is None:
                raise ValueError('complete box missing four coordinates')
            complete.append({'start': row_start, 'end': j,
                             'tokens': tuple([*prefix, *emitted][row_start:j + 1]), 'bins': coords})
            coords, row_start = [], None
    if phase != cell['final_partial_stage']:
        raise ValueError('final partial state changed')
    prior = [r for r in complete if r['end'] < len(prefix)]
    released = [r for r in complete if r['end'] >= len(prefix)]
    seen_prior = {r['tokens'] for r in prior}
    seen_release = set()
    repeat_prior = repeat_release = 0
    for r in released:
        a, b, c, d = r['bins']
        if not (a < c and b < d):
            raise ValueError('complete emitted box violates strict order before parser drops')
        repeat_prior += r['tokens'] in seen_prior
        repeat_release += r['tokens'] in seen_release
        seen_release.add(r['tokens'])
    metrics = cell['metrics']
    if (metrics['complete_row_count'] != len(released) or metrics['valid_geometry_row_count'] != len(released)
            or metrics['repeat_prior_prefix_count'] != repeat_prior
            or metrics['repeat_within_release_count'] != repeat_release
            or metrics['parser_drop_count'] != len(metrics['parser_drops'])):
        raise ValueError('saved raw row/repetition/parse denominator changed')
    return {'case': case['image_id'], 'steps': len(steps), 'complete_rows': len(released),
            'strictly_valid_rows': len(released), 'repeat_prior_prefix': repeat_prior,
            'repeat_within_release': repeat_release, 'final_stage': phase}


def verify(plan_path: Path, out: Path) -> Path:
    plan = json.loads(plan_path.read_text())
    if plan['schema'] != 'sustained_coordinate_legality.plan.v1' or len(plan['cases']) != 2:
        raise ValueError('frozen verification panel changed')
    rows = []
    for case in plan['cases']:
        native_path = Path(case['native_cell']['path'])
        if probe.binding(native_path)['sha256'] != case['native_cell']['sha256']:
            raise ValueError('reused native cell mutated')
        cell_path = study.ROOT / f'cells/{case["image_id"]}-sustained.json'
        rows.append(verify_cell(case, json.loads(cell_path.read_text()), plan['coordinate_token_ids']))
    probe.write_json(out, {'schema': 'sustained_coordinate_legality.saved_verification.v1',
                           'plan': probe.binding(plan_path), 'rows': rows, 'planned_new_cells': 2,
                           'complete_new_cells': len(rows), 'reused_native_cells': 2})
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, default=study.ROOT / 'selection/plan-v1.json')
    parser.add_argument('--out', type=Path, default=study.ROOT / 'qualification/saved-verification-v1.json')
    args = parser.parse_args()
    print(verify(args.plan, args.out))


if __name__ == '__main__':
    main()
