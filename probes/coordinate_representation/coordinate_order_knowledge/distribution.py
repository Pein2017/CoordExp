"""Saved-logit concentration and fixed-prefix source trajectory readout."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path

import torch

from probes.coordinate_representation.coordinate_order_knowledge import probe
from src.inference.bound_requests import build_bound_native_requests
from src.qwen.native import exact_history_inputs, prepare_native_inputs


ROOT = Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-23-coordinate-order-distribution')
OLD_PLAN = probe.ROOT / 'selection/plan-v1.json'
OLD_ACCEPTANCE = probe.ROOT / 'lead-acceptance-v1.json'
OLD_MANIFEST = probe.ROOT / 'candidate-v2/manifest.json'
LIMIT_FORWARDS = 256
LIMIT_SECONDS = 3600
ADMIT_SECONDS = 3300
LIMIT_BYTES = 1024**3
ROLES = {632: 'y2', 885: 'x2', 5586: 'x2', 7281: 'x2', 18380: 'x2'}


def metrics(read: dict, threshold: int, emitted: int, fixed_threshold: int | None = None,
            reference_bin: int | None = None) -> dict:
    """All masses are explicit about coordinate, legal, or full-vocabulary conditioning."""
    if not 0 <= threshold <= 999 or not 0 <= emitted <= 999:
        raise ValueError('invalid threshold or emitted bin')
    lc = read['coordinate_logprobs_conditional']
    lf = read['coordinate_logprobs_full_vocabulary']
    if len(lc) != 1000 or len(lf) != 1000:
        raise ValueError('incomplete coordinate family')
    p = [math.exp(x) for x in lc]
    pf = [math.exp(x) for x in lf]
    family = sum(pf)
    if abs(sum(p) - 1) > 2e-4 or abs(family - read['coordinate_family_mass']) > 2e-4:
        raise ValueError('coordinate normalization failed')
    illegal = list(range(threshold + 1))
    legal = list(range(threshold + 1, 1000))
    li = max(illegal, key=lambda k: lc[k])
    lk = max(legal, key=lambda k: lc[k]) if legal else None
    legal_mass = sum(p[k] for k in legal)
    conditional = {k: p[k] / legal_mass for k in legal} if legal_mass else {}
    order = sorted(legal, key=lambda k: (-conditional[k], k))
    entropy = -sum(v * math.log(v) for v in conditional.values() if v)
    quantiles = {}
    if legal_mass:
        cumulative = 0.
        next_q = iter((.05, .25, .5, .75, .95))
        wanted = next(next_q, None)
        for k in legal:
            cumulative += conditional[k]
            while wanted is not None and cumulative + 1e-12 >= wanted:
                quantiles[str(int(wanted * 100))] = k
                wanted = next(next_q, None)
    fixed = threshold if fixed_threshold is None else fixed_threshold
    if not 0 <= fixed <= 999:
        raise ValueError('invalid fixed threshold')
    return {
        'coordinate_family_mass_full_vocabulary': family,
        'illegal_mass_conditional_coordinate': sum(p[:threshold + 1]),
        'illegal_mass_full_vocabulary': sum(pf[:threshold + 1]),
        'fixed_partition_threshold': fixed,
        'fixed_illegal_mass_conditional_coordinate': sum(p[:fixed + 1]),
        'fixed_illegal_mass_full_vocabulary': sum(pf[:fixed + 1]),
        'best_illegal_bin': li, 'best_legal_bin': lk,
        'illegal_minus_legal_logprob_margin': None if lk is None else lc[li] - lc[lk],
        'legal_support_bins': len(legal),
        'legal_entropy_nats': entropy if len(legal) > 1 and legal_mass else None,
        'legal_entropy_normalized': entropy / math.log(len(legal)) if len(legal) > 1 and legal_mass else None,
        'legal_exp_entropy': math.exp(entropy) if len(legal) > 1 and legal_mass else None,
        'legal_top_mass': {str(n): sum(conditional[k] for k in order[:n]) if legal_mass else None
                           for n in (1, 5, 10, 50)},
        'best_legal_mass_full_vocabulary': pf[lk] if lk is not None else None,
        'legal_top_bins': order[:10],
        'legal_quantile_bins': quantiles,
        'legal_mass_near_top_bin': {str(radius): sum(conditional[k] for k in legal
                                                    if lk is not None and abs(k - lk) <= radius)
                                    if legal_mass else None for radius in (1, 5, 20)},
        'emitted_bin': emitted, 'emitted_legal_now': emitted > threshold,
        'emitted_probability_conditional_coordinate': p[emitted],
        'emitted_probability_full_vocabulary': pf[emitted],
        'emitted_logprob_full_vocabulary': lf[emitted],
        'emitted_rank_coordinate': 1 + sum(x > lc[emitted] for x in lc),
        'reference_bin': reference_bin,
        'reference_bin_legal_now': None if reference_bin is None else reference_bin > threshold,
        'reference_logprob_full_vocabulary': None if reference_bin is None else lf[reference_bin],
        'reference_probability_full_vocabulary': None if reference_bin is None else pf[reference_bin],
    }


def _old_cell(image_id: int, query: dict, prompt_sha: str) -> tuple[Path, dict] | None:
    name = f"{image_id}-{query['kind']}-{query['role']}-{query['query_index']}.json"
    path = probe.ROOT / 'cells' / name
    if not path.is_file():
        return None
    cell = json.loads(path.read_text())
    if cell['prompt_sha256'] != prompt_sha or cell['query']['prefix_sha256'] != query['prefix_sha256']:
        raise ValueError(f'old cell prompt/prefix conflict: {path}')
    base = [x for x in cell['conditions'] if x['axis'] == 'same' and
            x['requested_threshold'] == query['original_threshold']]
    if len(base) != 1 or base[0]['observed_threshold'] != query['original_threshold']:
        raise ValueError(f'old original condition conflict: {path}')
    return path, base[0]['readout']


def _window_queries(case: dict, tokens: list[dict], role: str, drops: list[dict]) -> tuple[list[dict], list[dict]]:
    all_boxes = probe.boxes(tokens)
    kind = 'invalid_native' if case['status'] == 'source_invalid' else 'healthy_native'
    original = next(q for q in case['queries'] if q['kind'] == kind and q['role'] == role)
    boundary_start = original['query_index'] - (3 if role == 'x2' else 4)
    matches = [i for i, b in enumerate(all_boxes) if b['box_start'] == boundary_start]
    if len(matches) != 1:
        raise ValueError('frozen boundary absent from generated complete boxes')
    boundary = matches[0]
    missing = []
    result = []
    ids = [int(t['token_id']) for t in tokens]
    chars = [0]
    for token in tokens:
        chars.append(chars[-1] + len(token['token_text']))
    for offset in range(-4, 9):
        index = boundary + offset
        if not 0 <= index < len(all_boxes):
            missing.append({'relative_row': offset, 'status': 'HOLD', 'reason': 'no complete saved row before EOS/cap'})
            continue
        box = all_boxes[index]
        q = next(x for x in probe.case_queries(tokens, box, 'trajectory') if x['role'] == role)
        q['history_token_ids'] = ids
        probe.validate_query(q)
        q['saved_selected_logprob'] = tokens[q['query_index']]['logprob']
        q['relative_row'] = offset
        q['complete_row_index'] = index
        q['box_start'] = box['box_start']
        q['exact_row_repeat'] = any(b['description'] == box['description'] and
                                    b['coord_bins'] == box['coord_bins'] for b in all_boxes[:index])
        q['malformed_history'] = any(d['reason'] != 'geometry_invalid' and
                                     d['char_end'] <= chars[q['query_index']] for d in drops)
        q['prior_invalid_geometry'] = any(b['invalid_x'] or b['invalid_y'] for b in all_boxes[:index])
        q['row_invalid_geometry'] = box['invalid_x'] or box['invalid_y']
        q.pop('history_token_ids')
        q.pop('thresholds')
        result.append(q)
    return result, missing


def prepare() -> Path:
    if ROOT.exists() and any(ROOT.rglob('*')):
        raise FileExistsError('distribution output already has prepared artifacts')
    old = json.loads(OLD_PLAN.read_text())
    if len(old['cases']) != 12 or len([q for c in old['cases'] for q in
       [*c['queries'], *(t for t in c['teacher'] if t.get('kind') == 'teacher')]]) != 70:
        raise ValueError('accepted 70-query denominator changed')
    traces = defaultdict(list)
    selected_ids = {c['image_id'] for c in old['cases']}
    for t in probe.jsonl(probe.SOURCE / 'pred_token_trace.jsonl'):
        if (t['trace_type'] == 'generated_token' and not t['is_pad'] and
                int(t['row_id'][-12:]) in selected_ids):
            traces[t['row_id']].append(t)
    for tokens in traces.values():
        tokens.sort(key=lambda t: t['generated_step_index'])
        if [t['generated_step_index'] for t in tokens] != list(range(len(tokens))):
            raise ValueError('noncontiguous source trace')
    drops = {x['row_id']: x['dropped_predictions'] for x in probe.jsonl(probe.SOURCE / 'parse_diagnostics.jsonl')}
    items, holds, reuse = [], [], []
    for case in old['cases']:
        image_id = case['image_id']
        role = ROLES[image_id] if image_id in ROLES else ROLES[int(case['match']['matched_invalid'][-12:])]
        tokens = traces[case['row_id']]
        queries, missing = _window_queries(case, tokens, role, drops[case['row_id']])
        holds += [{'image_id': image_id, **x} for x in missing]
        boundary = next(q for q in queries if q['relative_row'] == 0)
        for q in queries:
            saved = next((o for o in case['queries'] if o['role'] == role and
                         o['query_index'] == q['query_index'] and
                         o['prefix_sha256'] == q['prefix_sha256']), None)
            cell = _old_cell(image_id, saved, case['prompt_sha256']) if saved else None
            if cell:
                reuse.append(probe.binding(cell[0]))
            items.append({'image_id': image_id, 'role': role, 'status': case['status'],
                          'query': q, 'old_cell': str(cell[0]) if cell else None,
                          'prompt_sha256': case['prompt_sha256'],
                          'fixed_threshold': boundary['original_threshold'],
                          'reference_bin': boundary['observed_successor'] if case['status'] == 'source_invalid' else None})
    new_count = sum(x['old_cell'] is None for x in items)
    if new_count + 1 > LIMIT_FORWARDS:
        raise ValueError(f'planned {new_count}+1 forwards exceeds frozen {LIMIT_FORWARDS}')
    binds = [probe.binding(p) for p in (OLD_PLAN, OLD_MANIFEST, OLD_ACCEPTANCE,
              probe.SOURCE / 'pred_token_trace.jsonl', probe.SOURCE / 'parse_diagnostics.jsonl',
              probe.SOURCE / 'configs/resolved.json', Path(__file__), Path(probe.__file__))]
    binds.extend(sorted(reuse, key=lambda x: x['path']))
    plan = {'schema': 'coordinate_order_distribution.plan.v1', 'source_commit': probe.SOURCE_COMMIT,
            'source_runtime': old['source_runtime'], 'coordinate_token_ids': old['coordinate_token_ids'],
            'input_bindings': binds, 'old_plan_sha256': probe.binding(OLD_PLAN)['sha256'],
            'cases': [{'image_id': c['image_id'], 'row_id': c['row_id'], 'status': c['status'],
                       'match': c['match'], 'prompt_sha256': c['prompt_sha256'],
                       'media_sha256': c['media_sha256'], 'grid_thw': c['grid_thw']}
                      for c in old['cases']],
            'items': items, 'missing_window_rows': holds,
            'counts': {'cases': 12, 'stage_a_saved_queries': 70, 'trajectory_items': len(items),
                       'trajectory_reused': len(reuse), 'trajectory_new': new_count,
                       'window_hold_rows': len(holds), 'qualification_repeat_forwards': 1},
            'limits': {'new_forwards': LIMIT_FORWARDS, 'wall_seconds': LIMIT_SECONDS,
                       'gpu_seconds': LIMIT_SECONDS, 'admission_seconds': ADMIT_SECONDS,
                       'artifact_bytes': LIMIT_BYTES}}
    path = ROOT / 'selection/plan-v1.json'
    probe.write_json(path, plan)
    return path


def _read_item(item: dict) -> dict:
    q = item['query']
    if item['old_cell']:
        cell = json.loads(Path(item['old_cell']).read_text())
        if cell['prompt_sha256'] != item['prompt_sha256'] or cell['query']['prefix_sha256'] != q['prefix_sha256']:
            raise ValueError('reused cell identity changed')
        base = [x for x in cell['conditions'] if x['axis'] == 'same' and
                x['requested_threshold'] == q['original_threshold']]
        if len(base) != 1:
            raise ValueError('reused original condition changed')
        return base[0]['readout']
    path = ROOT / 'cells' / f"{item['image_id']}-{q['role']}-{q['query_index']}.json"
    d = json.loads(path.read_text())
    if d['prefix_sha256'] != q['prefix_sha256'] or d['threshold'] != q['original_threshold']:
        raise ValueError('new trajectory cell identity changed')
    return d['readout']


def stage_a() -> Path:
    old = json.loads(OLD_PLAN.read_text())
    rows = []
    for case in old['cases']:
        for q in [*case['queries'], *(t for t in case['teacher'] if t.get('kind') == 'teacher')]:
            saved = _old_cell(case['image_id'], q, case['prompt_sha256'])
            if saved is None:
                raise FileNotFoundError('accepted Stage A cell missing')
            rows.append({'image_id': case['image_id'], 'status': case['status'],
                         'kind': q['kind'], 'role': q['role'], 'query_index': q['query_index'],
                         'threshold': q['original_threshold'], 'source_cell': probe.binding(saved[0]),
                         'metrics': metrics(saved[1], q['original_threshold'], q['observed_successor'])})
    if len(rows) != 70:
        raise ValueError('Stage A denominator changed')
    path = ROOT / 'reduction/stage-a-v1.json'
    probe.write_json(path, {'schema': 'coordinate_order_distribution.stage_a.v1',
                            'old_plan': probe.binding(OLD_PLAN), 'saved_cells': len(rows), 'rows': rows})
    return path


def plot_stage_a() -> list[Path]:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    old = json.loads(OLD_PLAN.read_text())
    cases = {c['image_id']: c for c in old['cases']}
    paths = []
    for image_id, role in ROLES.items():
        invalid = cases[image_id]
        matched = [c for c in old['cases'] if c['status'] == 'healthy_control' and
                   int(c['match']['matched_invalid'][-12:]) == image_id]
        series = [(invalid, 'invalid_native'), (invalid, 'prior_valid_native'),
                  *((c, 'healthy_native') for c in matched)]
        fig, axes = plt.subplots(1, len(series), figsize=(5 * len(series), 3.4), sharey=True)
        for ax, (case, kind) in zip(axes, series):
            q = next(x for x in case['queries'] if x['kind'] == kind and x['role'] == role)
            saved = _old_cell(case['image_id'], q, case['prompt_sha256'])
            assert saved is not None
            read = saved[1]
            ys = [max(-12., x / math.log(10)) for x in read['coordinate_logprobs_conditional']]
            ax.plot(range(1000), ys, lw=.75)
            ax.axvline(q['original_threshold'], color='red', lw=1, label='threshold')
            ax.axvline(q['observed_successor'], color='black', lw=1, ls='--', label='emitted')
            ax.set_xlim(0, 999)
            ax.set_ylim(-12, .5)
            ax.set_xlabel('coordinate bin')
            ax.set_title(f"{case['image_id']} {kind} {role}\nthreshold {q['original_threshold']}, emitted {q['observed_successor']}")
        axes[0].set_ylabel('log10 P(bin | coordinate family), floor -12')
        axes[-1].legend(loc='lower right')
        fig.tight_layout()
        path = ROOT / 'plots' / f'{image_id}-{role}.png'
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=130)
        plt.close(fig)
        paths.append(path)
    return paths


def finalize_plan() -> Path:
    v1_path = ROOT / 'selection/plan-v1.json'
    plan = json.loads(v1_path.read_text())
    stage = ROOT / 'reduction/stage-a-v1.json'
    stage_rows = json.loads(stage.read_text())['rows']
    if len(stage_rows) != 70:
        raise ValueError('Stage A incomplete')
    old_module = Path(__file__).resolve()
    plan['input_bindings'] = [b for b in plan['input_bindings'] if b['path'] != str(old_module)]
    plan['input_bindings'] += [probe.binding(old_module), probe.binding(stage),
                              *(r['source_cell'] for r in stage_rows),
                              *(probe.binding(p) for p in sorted((ROOT / 'plots').glob('*.png')))]
    plan['prior_plan'] = probe.binding(v1_path)
    plan['schema'] = 'coordinate_order_distribution.plan.v2'
    path = ROOT / 'selection/plan-v2.json'
    probe.write_json(path, plan)
    return path


def run(plan_path: Path, device: str = 'cuda:0') -> Path:
    from src.artifacts.source_provenance import preserve_source
    plan = json.loads(plan_path.read_text())
    for saved in plan['input_bindings']:
        if probe.binding(Path(saved['path']))['sha256'] != saved['sha256']:
            raise ValueError(f"bound input changed: {saved['path']}")
    captures = []
    for relative in ('probes/coordinate_representation/coordinate_order_knowledge/distribution.py',
                     'probes/coordinate_representation/coordinate_order_knowledge/probe.py',
                     'src/qwen/native.py', 'src/inference/bound_requests.py',
                     'src/inference/hf_backend.py', 'src/adapters/dora.py'):
        path = Path.cwd() / relative
        captured = preserve_source(path, run_root=ROOT, relative_name=relative)
        captures.append({'current': probe.binding(path), 'capture': probe.binding(captured)})
    start = time.time()
    launch = ROOT / 'execution/launch-v1.json'
    probe.write_json(launch, {'plan': probe.binding(plan_path), 'captures': captures,
                              'device': device, 'pid': os.getpid(), 'model_wall_start': start,
                              'argv': list(__import__('sys').argv)})
    forwards = 0
    status, error = 'running', None
    try:
        config = json.loads((probe.SOURCE / 'configs/resolved.json').read_text())['config']
        qwen = probe._load_source(config, device)
        if list(qwen.token_identity.coordinate_token_ids) != plan['coordinate_token_ids']:
            raise ValueError('loaded coordinate IDs changed')
        old = json.loads(OLD_PLAN.read_text())
        cases = {c['image_id']: c for c in old['cases']}
        data = {r['image_id']: (i, r) for i, r in enumerate(probe.jsonl(probe.DATA))}
        media = {r['row_id']: r for r in probe.jsonl(probe.SOURCE / 'image_plan.jsonl')}
        first = True
        for case in plan['cases']:
            group = [x for x in plan['items'] if x['image_id'] == case['image_id'] and x['old_cell'] is None]
            if not group:
                continue
            if time.time() - start >= ADMIT_SECONDS:
                raise TimeoutError('new model admission cutoff')
            original = cases[case['image_id']]
            index, record = data[case['image_id']]
            mp = media[case['row_id']]
            row = {'row_id': case['row_id'], 'row_index': index, 'input_record': record,
                   'image_path': mp['image_path'], 'image_width': mp['decoded_width'],
                   'image_height': mp['decoded_height'], 'image_plan': mp}
            requests, _ = build_bound_native_requests(qwen, config, [row])
            batch = prepare_native_inputs(qwen.processor, requests, device=device, record_media_identity=True)
            prompt = list(batch.prompt_token_ids[0])
            if probe.digest_json(prompt) != case['prompt_sha256'] or list(batch.media_sha256 or ()) != [case['media_sha256']] or list(batch.image_grids[0]) != case['grid_thw']:
                raise ValueError('prompt, media or grid mismatch')
            for item in group:
                if time.time() - start >= ADMIT_SECONDS or forwards + 1 + int(first) > LIMIT_FORWARDS:
                    raise TimeoutError('wall or forward admission cutoff')
                q = item['query']
                history = [*prompt, *q['prefix_token_ids']]
                kwargs = exact_history_inputs(qwen.model, batch.inputs, [history], pad_token_id=0, logits_to_keep=1)
                actual = kwargs['input_ids'][0][kwargs['attention_mask'][0].bool()].tolist()
                if actual != history or kwargs['logits_to_keep'] != 1:
                    raise ValueError('causal history input changed')
                with torch.no_grad():
                    logits = qwen.model(**kwargs).logits[0, -1].float()
                forwards += 1
                if first:
                    with torch.no_grad():
                        replay = qwen.model(**kwargs).logits[0, -1].float()
                    forwards += 1
                    maximum = float((logits - replay).abs().max())
                    selected = float(torch.log_softmax(logits, 0)[q['observed_token_id']])
                    diff = abs(selected - q['saved_selected_logprob'])
                    qual = {'image_id': item['image_id'], 'query_index': q['query_index'],
                            'full_vocabulary_identity_max': maximum,
                            'saved_selected_logprob_difference': diff,
                            'greedy_equal_saved': int(logits.argmax()) == q['observed_token_id'],
                            'tolerance': 2e-4, 'forwards': forwards}
                    probe.write_json(ROOT / 'qualification/native-v1.json', qual)
                    if maximum > 2e-4 or diff > 2e-4 or not qual['greedy_equal_saved']:
                        raise ValueError('new-prefix identity or saved selected-token replay failed')
                    first = False
                read = probe.readout(logits, plan['coordinate_token_ids'], q['original_threshold'])
                probe.write_json(ROOT / 'cells' / f"{item['image_id']}-{q['role']}-{q['query_index']}.json",
                                 {'schema': 'coordinate_order_distribution.cell.v1',
                                  'image_id': item['image_id'], 'role': q['role'],
                                  'query_index': q['query_index'], 'prefix_sha256': q['prefix_sha256'],
                                  'threshold': q['original_threshold'], 'selected_logprob':
                                  float(torch.log_softmax(logits, 0)[q['observed_token_id']]),
                                  'readout': read})
                if sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()) >= LIMIT_BYTES:
                    raise RuntimeError('artifact-byte cap')
        status = 'complete'
    except Exception as exc:
        status, error = 'HOLD', repr(exc)
        raise
    finally:
        end = time.time()
        probe.write_json(ROOT / 'execution/terminal-v1.json',
                         {'status': status, 'error': error, 'pid': os.getpid(),
                          'wall_start': start, 'wall_end': end, 'allocated_gpu_seconds': end - start,
                          'new_forwards_including_qualification': forwards,
                          'gpu_count': 1})
    return ROOT / 'execution/terminal-v1.json'


def reduce(plan_path: Path) -> Path:
    plan = json.loads(plan_path.read_text())
    stage_a = json.loads((ROOT / 'reduction/stage-a-v1.json').read_text())['rows']
    trajectory = []
    missing = []
    for item in plan['items']:
        try:
            read = _read_item(item)
        except FileNotFoundError:
            missing.append({'image_id': item['image_id'], 'role': item['role'],
                            'query_index': item['query']['query_index'], 'status': 'HOLD'})
            continue
        q = item['query']
        trajectory.append({'image_id': item['image_id'], 'role': item['role'],
                           'status': item['status'], 'relative_row': q['relative_row'],
                           'complete_row_index': q['complete_row_index'],
                           'query_index': q['query_index'], 'prefix_sha256': q['prefix_sha256'],
                           'threshold': q['original_threshold'],
                           'fixed_threshold': item['fixed_threshold'],
                           'exact_row_repeat': q['exact_row_repeat'],
                           'malformed_history': q['malformed_history'],
                           'prior_invalid_geometry': q['prior_invalid_geometry'],
                           'row_invalid_geometry': q['row_invalid_geometry'],
                           'cell_path': item['old_cell'] or str(ROOT / 'cells' / f"{item['image_id']}-{q['role']}-{q['query_index']}.json"),
                           'metrics': metrics(read, q['original_threshold'], q['observed_successor'],
                                              item['fixed_threshold'], item['reference_bin'])})
    if len(stage_a) != 70:
        raise ValueError('Stage A denominator changed')
    result = {'schema': 'coordinate_order_distribution.reduction.v1', 'plan': probe.binding(plan_path),
              'counts': {'stage_a_saved': len(stage_a), 'trajectory_planned': len(plan['items']),
                         'trajectory_complete': len(trajectory), 'trajectory_cell_hold': len(missing),
                         'missing_window_hold': len(plan['missing_window_rows']),
                         'trajectory_reused': plan['counts']['trajectory_reused'],
                         'trajectory_new': plan['counts']['trajectory_new']},
              'stage_a': stage_a, 'trajectories': trajectory,
              'cell_holds': missing, 'window_holds': plan['missing_window_rows']}
    path = ROOT / 'reduction/reduction-v1.json'
    probe.write_json(path, result)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'stage-a', 'plots', 'finalize', 'run', 'reduce'))
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.action == 'prepare':
        print(prepare())
    elif args.action == 'stage-a':
        print(stage_a())
    elif args.action == 'plots':
        for path in plot_stage_a():
            print(path)
    elif args.action == 'finalize':
        print(finalize_plan())
    elif args.action == 'run':
        print(run(ROOT / 'selection/plan-v2.json', args.device))
    else:
        print(reduce(ROOT / 'selection/plan-v2.json'))


if __name__ == '__main__':
    main()
