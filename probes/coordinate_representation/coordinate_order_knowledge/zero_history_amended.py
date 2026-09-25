"""Frozen x2-zero versus same-row y1-zero history-content readout."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path

import torch

from probes.coordinate_representation.coordinate_order_knowledge import distribution, probe, zero_history
from src.inference.bound_requests import build_bound_native_requests
from src.qwen.native import exact_history_inputs, prepare_native_inputs


ROOT = zero_history.ROOT
PLAN = ROOT / 'selection/plan-v2.json'
OLD_DISTRIBUTION = distribution.ROOT / 'selection/plan-v2.json'
OLD_KNOWLEDGE = probe.ROOT / 'selection/plan-v1.json'
MAX_FORWARDS = 1024
PACKAGE_WALL_SECONDS = 7200
SHARED_WALL_SECONDS = 21600
ADMIT_SECONDS = 6900
MAX_ARTIFACT_BYTES = 2 * 1024**3


def eligible_rows(boxes: list[dict], current_box_start: int) -> list[dict]:
    return [{'box_start': b['box_start'], 'bins': b['coord_bins'],
             'y1_index': b['box_start'] + 2, 'x2_index': b['box_start'] + 3}
            for b in boxes if b['box_start'] < current_box_start and
            b['coord_bins'][1] == b['coord_bins'][2] == 0]


def _make_conditions(query: dict, rows: list[dict], prompt: list[int], replacement: int) -> list[dict]:
    """One replacement value; native is added once by the caller."""
    if not rows or replacement not in (1, 47):
        raise ValueError('no eligible historical rows or unsupported replacement')
    prefix = query['prefix_token_ids']
    conditions = []
    for subset, selected in (('latest', rows[-1:]), ('all', rows)):
        for role in ('x2', 'y1'):
            indices = [r[f'{role}_index'] for r in selected]
            if len(set(indices)) != len(selected) or any(i >= query['box_start'] for i in indices):
                raise ValueError('current-row or duplicate history edit')
            if any(prefix[i] != probe.COORD0 for i in indices):
                raise ValueError('history edit is not zero in the declared role')
            changed = prefix.copy()
            for i in indices:
                changed[i] = probe.COORD0 + replacement
            conditions.append({'name': f'{role}_{subset}_{replacement}', 'role': role,
                               'subset': subset, 'replacement': replacement,
                               'row_starts': [r['box_start'] for r in selected],
                               'indices': indices, 'edited_prefix_sha256': probe.digest_json(changed),
                               'full_history_sha256': probe.digest_json([*prompt, *changed])})
    return conditions


def validate_conditions(query: dict, rows: list[dict], prompt: list[int], conditions: list[dict]) -> None:
    """Fail closed on wrong role, row/count mismatch, current-row edits or unfrozen bytes."""
    if query['role'] != 'x2' or query['original_threshold'] != 0 or not rows:
        raise ValueError('only frozen current x1-zero x2 queries are admitted')
    if len(conditions) != 9 or [c['name'] for c in conditions] != [
            'native', 'x2_latest_1', 'y1_latest_1', 'x2_all_1', 'y1_all_1',
            'x2_latest_47', 'y1_latest_47', 'x2_all_47', 'y1_all_47']:
        raise ValueError('frozen nine-condition set changed')
    prefix = query['prefix_token_ids']
    if conditions[0]['edited_prefix_sha256'] != probe.digest_json(prefix) or conditions[0]['indices']:
        raise ValueError('native condition changed')
    for c in conditions[1:]:
        if c['role'] not in ('x2', 'y1') or c['subset'] not in ('latest', 'all') or c['replacement'] not in (1, 47):
            raise ValueError('unsupported role, subset or replacement')
        selected = rows[-1:] if c['subset'] == 'latest' else rows
        expected = [r[f"{c['role']}_index"] for r in selected]
        if (c['row_starts'] != [r['box_start'] for r in selected] or c['indices'] != expected or
                len(c['indices']) != len(selected) or any(i >= query['box_start'] for i in expected)):
            raise ValueError('role, same-row count or prior-only edit mismatch')
        changed = prefix.copy()
        for index in expected:
            if prefix[index] != probe.COORD0:
                raise ValueError('changed token was not an eligible zero')
            changed[index] = probe.COORD0 + c['replacement']
        if (c['edited_prefix_sha256'] != probe.digest_json(changed) or
                c['full_history_sha256'] != probe.digest_json([*prompt, *changed]) or
                changed[query['box_start']:] != prefix[query['box_start']:]):
            raise ValueError('prefix hash, full history or current-row mismatch')
    for subset in ('latest', 'all'):
        for replacement in (1, 47):
            pair = [c for c in conditions if c.get('subset') == subset and
                    c.get('replacement') == replacement]
            if (len(pair) != 2 or pair[0]['row_starts'] != pair[1]['row_starts'] or
                    len(pair[0]['indices']) != len(pair[1]['indices'])):
                raise ValueError('same-row paired count mismatch')


def _edited_prefix(query: dict, condition: dict) -> list[int]:
    prefix = query['prefix_token_ids'].copy()
    for index in condition['indices']:
        prefix[index] = probe.COORD0 + condition['replacement']
    return prefix


def _source_trace() -> dict[int, list[dict]]:
    ids = {image for image, _ in zero_history.TARGETS}
    traces = defaultdict(list)
    for token in probe.jsonl(probe.SOURCE / 'pred_token_trace.jsonl'):
        if (token['trace_type'] == 'generated_token' and not token['is_pad'] and
                int(token['row_id'][-12:]) in ids):
            traces[int(token['row_id'][-12:])].append(token)
    for tokens in traces.values():
        tokens.sort(key=lambda t: t['generated_step_index'])
        if [t['generated_step_index'] for t in tokens] != list(range(len(tokens))):
            raise ValueError('source generated history is not contiguous')
    return traces


def prepare() -> Path:
    from src.artifacts.source_provenance import preserve_source

    old = json.loads(OLD_DISTRIBUTION.read_text())
    knowledge = json.loads(OLD_KNOWLEDGE.read_text())
    stage_a = json.loads((distribution.ROOT / 'reduction/stage-a-v1.json').read_text())
    v1 = json.loads((ROOT / 'selection/admission-v1.json').read_text())
    traces = _source_trace()
    cases = {c['image_id']: c for c in knowledge['cases']}
    targets = []
    for image_id, relative in zero_history.TARGETS:
        item = next(x for x in old['items'] if x['image_id'] == image_id and
                    x['role'] == 'x2' and x['query']['relative_row'] == relative)
        q = item['query']
        ids = [int(t['token_id']) for t in traces[image_id]]
        if (q['prefix_sha256'] != probe.digest_json(ids[:q['query_index']]) or
                q['query_index'] != q['box_start'] + 3 or
                ids[q['query_index']] != q['observed_token_id'] or
                ids[q['box_start'] + 1] != probe.COORD0):
            raise ValueError('accepted native causal query changed')
        rows = eligible_rows(probe.boxes(traces[image_id]), q['box_start'])
        if len(rows) != next(x['history']['x2_zero'] for x in v1['targets']
                             if x['image_id'] == image_id and x['relative_row'] == relative):
            raise ValueError('eligible historical row count differs from accepted CPU census')
        original = next(x for x in stage_a['rows'] if x['image_id'] == image_id and
                        x['kind'] == 'invalid_native' and x['role'] == 'x2')
        replacement = original['metrics']['best_legal_bin']
        if replacement != 47:
            raise ValueError('first-illegal frozen legal replacement changed')
        prompt = cases[image_id]['prompt_token_ids']
        native = {'name': 'native', 'role': None, 'subset': None, 'replacement': None,
                  'row_starts': [], 'indices': [],
                  'edited_prefix_sha256': q['prefix_sha256'],
                  'full_history_sha256': probe.digest_json([*prompt, *q['prefix_token_ids']])}
        conditions = [native, *_make_conditions(q, rows, prompt, 1),
                      *_make_conditions(q, rows, prompt, 47)]
        validate_conditions(q, rows, prompt, conditions)
        original_cell = distribution.ROOT / 'cells' / f"{image_id}-x2-{q['query_index']}.json"
        saved = json.loads(original_cell.read_text())
        if saved['prefix_sha256'] != q['prefix_sha256'] or saved['threshold'] != 0:
            raise ValueError('preintervention saved logit cell changed')
        logp = saved['readout']['coordinate_logprobs_conditional']
        frozen_best_legal = max(range(1, 1000), key=lambda k: logp[k])
        targets.append({'image_id': image_id, 'relative_row': relative,
                        'row_id': probe.row_id(image_id), 'query': q,
                        'prompt_token_ids': prompt, 'prompt_sha256': cases[image_id]['prompt_sha256'],
                        'media_sha256': cases[image_id]['media_sha256'],
                        'grid_thw': cases[image_id]['grid_thw'],
                        'eligible_rows': rows, 'conditions': conditions,
                        'first_illegal_best_legal_replacement': replacement,
                        'frozen_late_best_legal_bin': frozen_best_legal,
                        'saved_native_cell': probe.binding(original_cell)})
    capture_pairs = []
    for relative_path in ('probes/coordinate_representation/coordinate_order_knowledge/zero_history_amended.py',
                          'probes/coordinate_representation/coordinate_order_knowledge/probe.py',
                          'probes/coordinate_representation/coordinate_order_knowledge/distribution.py',
                          'src/qwen/native.py', 'src/inference/bound_requests.py',
                          'src/inference/hf_backend.py', 'src/adapters/dora.py'):
        source = Path.cwd() / relative_path
        capture = preserve_source(source, run_root=ROOT, relative_name=relative_path)
        capture_pairs.append({'current': probe.binding(source), 'capture': probe.binding(capture)})
    payload = [b for b in knowledge['source_bindings'] if Path(b['path']).name != 'probe.py']
    for b in payload:
        if probe.binding(Path(b['path']))['sha256'] != b['sha256']:
            raise ValueError(f"accepted source payload changed: {b['path']}")
    binds = [probe.binding(p) for p in (ROOT / 'candidate-v1/manifest.json',
             ROOT / 'selection/admission-v1.json', OLD_DISTRIBUTION, OLD_KNOWLEDGE,
             distribution.ROOT / 'lead-acceptance-v1.json',
             distribution.ROOT / 'reduction/stage-a-v1.json',
             probe.SOURCE / 'pred_token_trace.jsonl', probe.SOURCE / 'configs/resolved.json',
             probe.SOURCE / 'image_plan.jsonl', probe.DATA,
             Path.cwd() / 'research/experiments/2026-09-23-coordinate-zero-history-feedback/amendment-v2.md')]
    binds += [t['saved_native_cell'] for t in targets]
    plan = {'schema': 'coordinate_zero_history_feedback.plan.v2',
            'source_commit': probe.SOURCE_COMMIT,
            'source_runtime': knowledge['source_runtime'],
            'coordinate_token_ids': knowledge['coordinate_token_ids'],
            'targets': targets, 'healthy_controls': {'status': 'HOLD', 'eligible_queries': 0},
            'condition_count': sum(len(t['conditions']) for t in targets),
            'planned_new_model_forwards_including_identity': 37,
            'input_bindings': binds, 'accepted_payload_bindings': payload,
            'current_capture_pairs': capture_pairs,
            'limits': {'package_wall_seconds': PACKAGE_WALL_SECONDS,
                       'shared_wall_seconds': SHARED_WALL_SECONDS,
                       'package_gpu_seconds': 4 * 3600, 'shared_gpu_seconds': 16 * 3600,
                       'forwards': MAX_FORWARDS, 'artifact_bytes': MAX_ARTIFACT_BYTES,
                       'new_admission_seconds': ADMIT_SECONDS}}
    path = PLAN
    probe.write_json(path, plan)
    return path


def _readout(logits: torch.Tensor, coordinate_ids: list[int], frozen_best: int) -> dict:
    read = probe.readout(logits, coordinate_ids, 0)
    lc = read['coordinate_logprobs_conditional']
    lf = read['coordinate_logprobs_full_vocabulary']
    legal = max(range(1, 1000), key=lambda k: lc[k])
    read.update({'zero_probability_conditional_coordinate': math.exp(lc[0]),
                 'zero_probability_full_vocabulary': math.exp(lf[0]),
                 'best_legal_bin': legal,
                 'zero_minus_best_legal_logprob_margin': lc[0] - lc[legal],
                 'zero_minus_frozen_legal_logprob_margin': lc[0] - lc[frozen_best],
                 'frozen_best_legal_bin': frozen_best,
                 'frozen_best_legal_logprob_conditional_coordinate': lc[frozen_best],
                 'frozen_best_legal_logprob_full_vocabulary': lf[frozen_best],
                 'bin999_logprob_conditional_coordinate': lc[999],
                 'bin999_logprob_full_vocabulary': lf[999]})
    return read


def run(plan_path: Path, device: str = 'cuda:0') -> Path:
    plan = json.loads(plan_path.read_text())
    for b in [*plan['input_bindings'], *plan['accepted_payload_bindings']]:
        if probe.binding(Path(b['path']))['sha256'] != b['sha256']:
            raise ValueError(f"frozen input changed: {b['path']}")
    for pair in plan['current_capture_pairs']:
        if (probe.binding(Path(pair['current']['path']))['sha256'] != pair['current']['sha256'] or
                probe.binding(Path(pair['capture']['path']))['sha256'] != pair['capture']['sha256']):
            raise ValueError('current or captured executed source changed')
    clock_path = ROOT / 'execution/overnight-model-start-v2.txt'
    if not clock_path.is_file():
        raise FileNotFoundError('common model-entry clock must be written before process launch')
    common_start = float(clock_path.read_text())
    launch = ROOT / 'execution/launch-v2.json'
    probe.write_json(launch, {'plan': probe.binding(plan_path), 'device': device,
                              'pid': os.getpid(), 'common_model_wall_start': common_start,
                              'argv': list(__import__('sys').argv)})
    forwards = 0
    status, error = 'running', None
    start = time.time()
    try:
        if start - common_start >= ADMIT_SECONDS:
            raise TimeoutError('package admission cutoff before model load')
        config = json.loads((probe.SOURCE / 'configs/resolved.json').read_text())['config']
        qwen = probe._load_source(config, device)
        if list(qwen.token_identity.coordinate_token_ids) != plan['coordinate_token_ids']:
            raise ValueError('loaded coordinate IDs differ from frozen source')
        data = {r['image_id']: (i, r) for i, r in enumerate(probe.jsonl(probe.DATA))}
        media = {r['row_id']: r for r in probe.jsonl(probe.SOURCE / 'image_plan.jsonl')}
        first = True
        for target in plan['targets']:
            if time.time() - common_start >= ADMIT_SECONDS or forwards + len(target['conditions']) + int(first) > MAX_FORWARDS:
                raise TimeoutError('wall or forward admission cutoff')
            image_id, row_id = target['image_id'], target['row_id']
            index, record = data[image_id]
            mp = media[row_id]
            case = {'row_id': row_id, 'row_index': index, 'input_record': record,
                    'image_path': mp['image_path'], 'image_width': mp['decoded_width'],
                    'image_height': mp['decoded_height'], 'image_plan': mp}
            requests, _ = build_bound_native_requests(qwen, config, [case])
            batch = prepare_native_inputs(qwen.processor, requests, device=device, record_media_identity=True)
            prompt = list(batch.prompt_token_ids[0])
            if (prompt != target['prompt_token_ids'] or
                    list(batch.media_sha256 or ()) != [target['media_sha256']] or
                    list(batch.image_grids[0]) != target['grid_thw']):
                raise ValueError('prompt, image content or processed grid changed')
            q, rows = target['query'], target['eligible_rows']
            validate_conditions(q, rows, prompt, target['conditions'])
            original = [*prompt, *q['prefix_token_ids']]
            base_kwargs = exact_history_inputs(qwen.model, batch.inputs, [original], pad_token_id=0, logits_to_keep=1)
            for condition in target['conditions']:
                edited = _edited_prefix(q, condition)
                history = [*prompt, *edited]
                if probe.digest_json(history) != condition['full_history_sha256']:
                    raise ValueError('edited full history differs from frozen condition')
                kwargs = exact_history_inputs(qwen.model, batch.inputs, [history], pad_token_id=0, logits_to_keep=1)
                actual = kwargs['input_ids'][0][kwargs['attention_mask'][0].bool()].tolist()
                if actual != history or kwargs['logits_to_keep'] != 1:
                    raise ValueError('exact causal history changed')
                for key in ('attention_mask', 'position_ids', 'image_grid_thw', 'pixel_values'):
                    if key in base_kwargs or key in kwargs:
                        if key not in base_kwargs or key not in kwargs or not torch.equal(base_kwargs[key], kwargs[key]):
                            raise ValueError(f'{key} changed under token-only history edit')
                delta = [i for i, (a, b) in enumerate(zip(original, history)) if a != b]
                expected = [len(prompt) + i for i in condition['indices']]
                if len(original) != len(history) or delta != expected:
                    raise ValueError('actual-caller edited token positions changed')
                with torch.no_grad():
                    logits = qwen.model(**kwargs).logits[0, -1].float()
                forwards += 1
                if condition['name'] == 'native':
                    with torch.no_grad():
                        identity = qwen.model(**kwargs).logits[0, -1].float() if first else None
                    if first:
                        forwards += 1
                    selected = float(torch.log_softmax(logits, 0)[q['observed_token_id']])
                    saved_selected = q['saved_selected_logprob']
                    saved = json.loads(Path(target['saved_native_cell']['path']).read_text())['readout']
                    current = probe.readout(logits, plan['coordinate_token_ids'], 0)
                    maximum = max(abs(a - b) for a, b in zip(
                        current['coordinate_logprobs_conditional'], saved['coordinate_logprobs_conditional']))
                    qual = {'image_id': image_id, 'query_index': q['query_index'],
                            'selected_logprob_abs_diff': abs(selected - saved_selected),
                            'coordinate_conditional_logprob_max_diff': maximum,
                            'full_vocabulary_winner_equal_saved': int(logits.argmax()) == q['observed_token_id'],
                            'full_vocabulary_identity_max': float((logits - identity).abs().max()) if first else None,
                            'tolerance': 2e-4}
                    probe.write_json(ROOT / 'qualification' / f"{image_id}-{q['query_index']}-native-v2.json", qual)
                    if (qual['selected_logprob_abs_diff'] > 2e-4 or maximum > 2e-4 or
                            not qual['full_vocabulary_winner_equal_saved'] or
                            (first and qual['full_vocabulary_identity_max'] > 2e-4)):
                        raise ValueError('native replay/identity qualification failed')
                    first = False
                read = _readout(logits, plan['coordinate_token_ids'], target['frozen_late_best_legal_bin'])
                probe.write_json(ROOT / 'cells/v2' / f"{image_id}-{q['query_index']}-{condition['name']}.json",
                                 {'schema': 'coordinate_zero_history_feedback.cell.v2',
                                  'image_id': image_id, 'query_index': q['query_index'],
                                  'condition': condition['name'], 'full_history_sha256':
                                  condition['full_history_sha256'], 'readout': read})
                if sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()) >= MAX_ARTIFACT_BYTES:
                    raise RuntimeError('artifact-byte cap')
        status = 'complete'
    except Exception as exc:
        status, error = 'HOLD', repr(exc)
        raise
    finally:
        end = time.time()
        probe.write_json(ROOT / 'execution/terminal-v2.json',
                         {'status': status, 'error': error, 'pid': os.getpid(),
                          'common_model_wall_start': common_start, 'process_wall_end': end,
                          'allocated_gpu_seconds_conservative': end - common_start,
                          'new_forwards': forwards, 'gpu_count': 1})
    return ROOT / 'execution/terminal-v2.json'


def reduce(plan_path: Path, output: Path) -> Path:
    plan = json.loads(plan_path.read_text())
    cases = []
    for target in plan['targets']:
        cells = []
        for condition in target['conditions']:
            path = ROOT / 'cells/v2' / f"{target['image_id']}-{target['query']['query_index']}-{condition['name']}.json"
            cell = json.loads(path.read_text())
            if cell['condition'] != condition['name'] or cell['full_history_sha256'] != condition['full_history_sha256']:
                raise ValueError('cell identity differs from frozen condition')
            cells.append({'condition': condition, 'cell': probe.binding(path), 'readout': cell['readout']})
        native = cells[0]['readout']
        effects = []
        for c in cells[1:]:
            value = c['readout']
            effects.append({'condition': c['condition']['name'],
                            'role': c['condition']['role'], 'subset': c['condition']['subset'],
                            'replacement': c['condition']['replacement'],
                            'margin_delta_vs_native': value['zero_minus_best_legal_logprob_margin'] -
                            native['zero_minus_best_legal_logprob_margin'],
                            'frozen_legal_margin_delta_vs_native':
                            value['zero_minus_frozen_legal_logprob_margin'] -
                            native['zero_minus_frozen_legal_logprob_margin'],
                            'zero_full_probability_delta_vs_native':
                            value['zero_probability_full_vocabulary'] -
                            native['zero_probability_full_vocabulary'],
                            'full_vocabulary_winner': value['full_vocabulary_winner'],
                            'best_legal_bin': value['best_legal_bin']})
        pairs = []
        for replacement in (1, 47):
            for subset in ('latest', 'all'):
                x = next(e for e in effects if e['role'] == 'x2' and e['subset'] == subset and e['replacement'] == replacement)
                y = next(e for e in effects if e['role'] == 'y1' and e['subset'] == subset and e['replacement'] == replacement)
                pairs.append({'replacement': replacement, 'subset': subset,
                              'x2_minus_y1_margin_delta': x['margin_delta_vs_native'] -
                              y['margin_delta_vs_native'],
                              'x2_minus_y1_frozen_margin_delta': x['frozen_legal_margin_delta_vs_native'] -
                              y['frozen_legal_margin_delta_vs_native'],
                              'x2_minus_y1_zero_full_probability_delta':
                              x['zero_full_probability_delta_vs_native'] -
                              y['zero_full_probability_delta_vs_native']})
        cases.append({'image_id': target['image_id'], 'relative_row': target['relative_row'],
                      'query_index': target['query']['query_index'], 'edited_row_count_all': len(target['eligible_rows']),
                      'replacement_values': [1, 47], 'native': native, 'cells': cells,
                      'effects': effects, 'paired': pairs})
    result = {'schema': 'coordinate_zero_history_feedback.reduction.v2',
              'plan': probe.binding(plan_path),
              'counts': {'cases': len(cases), 'conditions': sum(len(c['cells']) for c in cases),
                         'healthy_controls': 0, 'healthy_status': 'HOLD',
                         'new_model_forwards_including_identity': 37},
              'cases': cases}
    probe.write_json(output, result)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'run', 'reduce'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if args.action == 'prepare':
        print(prepare())
    elif args.action == 'run':
        print(run(PLAN, args.device))
    else:
        print(reduce(PLAN, args.out or ROOT / 'reduction/reduction-v2.json'))


if __name__ == '__main__':
    main()
