"""Six fixed source continuations after the first illegal x2 choice."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

from probes.coordinate_representation.coordinate_order_knowledge import probe
from src.inference.bound_requests import build_bound_native_requests
from src.inference.parsing import parse_compact_object_box_closed
from src.qwen.generation import NativeGenerationPolicy, generate_continuations
from src.qwen.native import prepare_native_inputs


ROOT = Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-23-first-illegal-coordinate-release')
PRIOR = Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-23-coordinate-order-knowledge')
COMMON_START = 1790182797.5338483
PREVIOUS_GPU_SECONDS = 37.97752809524536
HORIZON = 384
EOS = 151645
POLICY = NativeGenerationPolicy(temperature=0, top_p=1, top_k=0, repetition_penalty=1, use_model_defaults=False)


def validate_boundary(prefix: list[int], query_index: int, token: int) -> None:
    if (len(prefix) != query_index or query_index < 3 or prefix[-3] != probe.BOX_START
            or prefix[-2:] != [probe.COORD0, probe.COORD0]
            or token not in (probe.COORD0, probe.COORD0 + 1, probe.COORD0 + 47)):
        raise ValueError('first illegal x2 boundary or intervention is wrong')


def check_unrestricted(winners: list[int], emitted: list[int]) -> None:
    if len(winners) != len(emitted) or any(a != b for a, b in zip(winners, emitted, strict=True)):
        raise ValueError('override persisted after the first token or greedy path changed')


def _source_trace(image_id: int) -> list[dict]:
    rows = [t for t in probe.jsonl(probe.SOURCE / 'pred_token_trace.jsonl')
            if t['row_id'] == probe.row_id(image_id) and t['trace_type'] == 'generated_token' and not t['is_pad']]
    rows.sort(key=lambda x: x['generated_step_index'])
    if [x['generated_step_index'] for x in rows] != list(range(len(rows))):
        raise ValueError('noncontiguous accepted source trace')
    return rows


def prepare() -> Path:
    old_path = PRIOR / 'selection/plan-v1.json'
    old = json.loads(old_path.read_text())
    cases = []
    bindings = [probe.binding(old_path), probe.binding(probe.SOURCE / 'pred_token_trace.jsonl'),
                probe.binding(probe.SOURCE / 'configs/resolved.json'), probe.binding(probe.SOURCE / 'image_plan.jsonl'),
                probe.binding(probe.DATA), probe.binding(Path(__file__))]
    for saved in old['source_bindings']:
        if Path(saved['path']).resolve() == Path(probe.__file__).resolve():
            continue  # historical preparation source binding is separately disclosed by that accepted unit
        if probe.binding(Path(saved['path']))['sha256'] != saved['sha256']:
            raise ValueError(f'accepted source binding changed: {saved["path"]}')
    source_cells = []
    images = {x['row_id']: x for x in probe.jsonl(probe.SOURCE / 'image_plan.jsonl')}
    for image_id in (885, 5586):
        case = next(c for c in old['cases'] if c['image_id'] == image_id)
        q = next(q for q in case['queries'] if q['kind'] == 'invalid_native' and q['role'] == 'x2')
        trace = _source_trace(image_id)
        ids = [int(t['token_id']) for t in trace]
        if q['prefix_token_ids'] != ids[:q['query_index']] or ids[q['query_index']] != probe.COORD0:
            raise ValueError('frozen first-illegal source prefix changed')
        validate_boundary(ids[:q['query_index']], q['query_index'], probe.COORD0)
        preceding = [b for b in probe.boxes(trace) if b['invalid_x'] and b['box_start'] < q['query_index'] - 3]
        if preceding:
            raise ValueError('selected x2 is not first illegal x2')
        next64 = ids[q['query_index']:q['query_index'] + 64]
        if len(next64) != 64:
            raise ValueError('accepted source lacks 64-token replay window')
        cell_path = PRIOR / f'cells/{image_id}-invalid_native-x2-{q["query_index"]}.json'
        cell = json.loads(cell_path.read_text())
        native = next(c for c in cell['conditions'] if c['axis'] == 'same' and c['requested_threshold'] == 0)
        coord_lp = native['readout']['coordinate_logprobs_full_vocabulary']
        best_legal = max(range(1, 1000), key=lambda i: coord_lp[i])
        if best_legal != 47 or native['readout']['full_vocabulary_winner'] != probe.COORD0:
            raise ValueError('frozen best-legal value or illegal winner changed')
        plan_row = images[case['row_id']]
        bindings += [probe.binding(cell_path), probe.binding(Path(plan_row['image_path']))]
        source_cells.append(probe.binding(cell_path))
        cases.append({'image_id': image_id, 'row_id': case['row_id'], 'query_index': q['query_index'],
                      'prefix_token_ids': q['prefix_token_ids'], 'prefix_sha256': q['prefix_sha256'],
                      'prompt_token_ids': case['prompt_token_ids'], 'media_sha256': case['media_sha256'],
                      'grid_thw': case['grid_thw'], 'saved_next64': next64,
                      'source_cell': probe.binding(cell_path), 'source_native_coordinate_logprobs_full': coord_lp,
                      'source_native_winner': native['readout']['full_vocabulary_winner'],
                      'source_native_selected_logprob': q['saved_selected_logprob'],
                      'image_plan': plan_row})
    plan = {'schema': 'first_illegal_coordinate_release.plan.v1', 'source_commit': old['source_commit'],
            'source_runtime': old['source_runtime'], 'source_bindings': old['source_bindings'],
            'coordinate_token_ids': old['coordinate_token_ids'],
            'bindings': bindings, 'source_cells': source_cells, 'cases': cases,
            'conditions': ['native', 'force47', 'force1'], 'horizon_including_first': HORIZON,
            'policy': {'temperature': 0, 'top_p': 1, 'top_k': 0, 'repetition_penalty': 1,
                       'use_model_defaults': False}, 'eos_token_id': EOS,
            'limits': {'package_wall_seconds': 5400, 'package_gpu_seconds': 7200,
                       'package_forwards': 4096, 'package_artifact_bytes': 1024**3,
                       'shared_start': COMMON_START, 'shared_gpu_seconds_before': PREVIOUS_GPU_SECONDS}}
    path = ROOT / 'selection/plan-v1.json'
    probe.write_json(path, plan)
    return path


def _rows(ids: list[int], tokenizer) -> list[dict]:
    rows = []
    for j, token in enumerate(ids):
        if token != probe.REF_START:
            continue
        try:
            end = ids.index(probe.REF_END, j + 1)
        except ValueError:
            continue
        if end + 6 >= len(ids) or ids[end + 1] != probe.BOX_START or ids[end + 6] != probe.BOX_END:
            continue
        coords = ids[end + 2:end + 6]
        if not all(probe.COORD0 <= x < probe.COORD0 + 1000 for x in coords):
            continue
        bins = [x - probe.COORD0 for x in coords]
        description = tokenizer.decode(ids[j + 1:end], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        rows.append({'start': j, 'end': end + 6, 'description': description, 'bins': bins,
                     'equality_x': bins[0] == bins[2], 'equality_y': bins[1] == bins[3],
                     'reversal_x': bins[0] > bins[2], 'reversal_y': bins[1] > bins[3]})
    return rows


def _analyze(prefix: list[int], release: list[int], tokenizer, case: dict, stop: str) -> dict:
    all_ids = [*prefix, *release]
    all_rows = _rows(all_ids, tokenizer)
    preceding = [r for r in all_rows if r['end'] < len(prefix)]
    current = [r for r in all_rows if r['end'] >= len(prefix)]
    key = lambda r: (r['description'], tuple(r['bins']))
    prior = {key(r): i for i, r in enumerate(preceding)}
    within = {}
    first_repeat = None
    max_run = run = 0
    last_key = key(preceding[-1]) if preceding else None
    annotated = []
    for i, row in enumerate(current):
        k = key(row)
        previous = within.get(k, prior.get(k))
        if first_repeat is None and previous is not None:
            first_repeat = {'release_row_1based': i + 1, 'distance_complete_rows': len(preceding) + i - previous,
                            'against': 'within_release' if k in within else 'prior_prefix'}
        annotated.append({**row, 'repeat_prior_prefix': k in prior, 'repeat_within_release': k in within})
        run = run + 1 if k == last_key else 1
        max_run = max(max_run, run)
        last_key = k
        within.setdefault(k, len(preceding) + i)
    raw_text = tokenizer.decode(all_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    boundary_char = len(tokenizer.decode(prefix, skip_special_tokens=False, clean_up_tokenization_spaces=False))
    parsed = parse_compact_object_box_closed(raw_text, row_id=case['row_id'], row_index=0,
                                              image_width=case['image_plan']['decoded_width'],
                                              image_height=case['image_plan']['decoded_height'])
    drops = [d for d in parsed.dropped_predictions if d.get('char_end', -1) > boundary_char]
    return {'complete_release_rows': annotated, 'complete_row_count': len(current),
            'valid_geometry_row_count': sum(not (r['equality_x'] or r['equality_y'] or r['reversal_x'] or r['reversal_y']) for r in current),
            'repeat_prior_prefix_count': sum(r['repeat_prior_prefix'] for r in annotated),
            'repeat_within_release_count': sum(r['repeat_within_release'] for r in annotated),
            'first_recurrence': first_repeat, 'max_contiguous_repeat_run': max_run,
            'invalid_equality_x': sum(r['equality_x'] for r in current),
            'invalid_equality_y': sum(r['equality_y'] for r in current),
            'invalid_reversal_x': sum(r['reversal_x'] for r in current),
            'invalid_reversal_y': sum(r['reversal_y'] for r in current),
            'parser_drops': drops, 'parser_drop_count': len(drops), 'parse_status': parsed.parse_status,
            'stop_reason': stop, 'release_length': len(release), 'cap': stop == 'length',
            'release_text': tokenizer.decode(release, skip_special_tokens=False, clean_up_tokenization_spaces=False)}


def _continue(qwen, batch, prefix: list[int], budget: int) -> tuple[object, dict]:
    model = qwen.model
    original = model.generate
    seen = []

    def capture(**kwargs):
        result = original(**kwargs)
        width = kwargs['input_ids'].shape[1]
        emitted = result.sequences[0, width:].tolist()
        logits = tuple(result.logits)
        winners = [int(z[0].argmax().item()) for z in logits]
        check_unrestricted(winners, emitted)
        seen.append({'first_logits': logits[0][0].float().cpu(), 'winners': winners,
                     'generated_steps': len(logits), 'input_width': width})
        return result

    model.generate = capture
    try:
        result = generate_continuations(model, batch, extensions=[prefix], budgets=[budget], eos_token_id=EOS,
                                        pad_token_id=qwen.tokenizer.pad_token_id, policy=POLICY,
                                        trace='raw_and_policy', seed=None)[0]
    finally:
        model.generate = original
    if len(seen) != 1 or seen[0]['generated_steps'] != len(result.token_ids):
        raise ValueError('cached generation trace was not one unpadded target trajectory')
    return result, seen[0]


def run(plan_path: Path, device: str = 'cuda:0') -> Path:
    from src.artifacts.source_provenance import preserve_source

    plan = json.loads(plan_path.read_text())
    if plan['schema'] != 'first_illegal_coordinate_release.plan.v1' or len(plan['cases']) != 2:
        raise ValueError('frozen plan changed')
    for b in plan['bindings']:
        if probe.binding(Path(b['path']))['sha256'] != b['sha256']:
            raise ValueError(f'input binding changed: {b["path"]}')
    for b in plan['source_bindings']:
        if Path(b['path']).resolve() != Path(probe.__file__).resolve() and probe.binding(Path(b['path']))['sha256'] != b['sha256']:
            raise ValueError(f'accepted source binding changed: {b["path"]}')
    capture_paths = ('probes/coordinate_representation/coordinate_order_knowledge/first_illegal_release.py',
                     'probes/coordinate_representation/coordinate_order_knowledge/probe.py',
                     'src/qwen/native.py', 'src/qwen/generation.py', 'src/inference/hf_backend.py',
                     'src/inference/parsing.py', 'src/inference/bound_requests.py')
    captures = []
    for rel in capture_paths:
        current = Path.cwd() / rel
        saved = preserve_source(current, run_root=ROOT, relative_name=rel)
        captures.append({'current': probe.binding(current), 'capture': probe.binding(saved)})
    start = time.time()
    launch = ROOT / 'execution/launch-v1.json'
    probe.write_json(launch, {'plan': probe.binding(plan_path), 'captures': captures, 'device': device,
                              'pid': os.getpid(), 'argv': list(__import__('sys').argv),
                              'package_model_start': start, 'shared_model_start': COMMON_START})
    config = json.loads((probe.SOURCE / 'configs/resolved.json').read_text())['config']
    forwards = 0
    status = 'failed'
    try:
        qwen = probe._load_source(config, device)
        images = {x['row_id']: x for x in probe.jsonl(probe.SOURCE / 'image_plan.jsonl')}
        dataset = {x['image_id']: x for x in probe.jsonl(probe.DATA)}
        for case in plan['cases']:
            if time.time() - start >= 5100 or time.time() - COMMON_START >= 5.75 * 3600:
                raise TimeoutError('package or shared admission reserve reached')
            image = case['image_id']
            row = {'row_id': case['row_id'], 'row_index': 0, 'input_record': dataset[image],
                   'image_path': images[case['row_id']]['image_path'],
                   'image_width': images[case['row_id']]['decoded_width'],
                   'image_height': images[case['row_id']]['decoded_height'],
                   'image_plan': images[case['row_id']]}
            requests, _ = build_bound_native_requests(qwen, config, [row])
            batch = prepare_native_inputs(qwen.processor, requests, device=device, record_media_identity=True)
            prompt = list(batch.prompt_token_ids[0])
            if (prompt != case['prompt_token_ids'] or list(batch.media_sha256 or ()) != [case['media_sha256']]
                    or list(batch.image_grids[0]) != case['grid_thw']):
                raise ValueError('source prompt/media/grid changed')
            prefix = case['prefix_token_ids']
            validate_boundary(prefix, case['query_index'], probe.COORD0)
            prefill = probe._native_logits(qwen, batch.inputs, [*prompt, *prefix], [0])
            forwards += 1
            old = case['source_native_coordinate_logprobs_full']
            new = probe.readout(prefill, plan['coordinate_token_ids'], 0)['coordinate_logprobs_full_vocabulary']
            prefill_max = max(abs(a - b) for a, b in zip(old, new, strict=True))
            if prefill_max > 2e-4 or int(prefill.argmax()) != probe.COORD0:
                raise ValueError(f'accepted source prefill mismatch: {image}, max={prefill_max}')
            condition_order = [('native', probe.COORD0), ('force47', probe.COORD0 + 47), ('force1', probe.COORD0 + 1)]
            for condition, first in condition_order:
                if time.time() - start >= 5100 or time.time() - COMMON_START >= 5.75 * 3600 or forwards + HORIZON > 4096:
                    raise TimeoutError('model admission/forward cap reached')
                validate_boundary(prefix, case['query_index'], first)
                if condition == 'native':
                    result, internal = _continue(qwen, batch, prefix, HORIZON)
                    release = list(result.token_ids)
                    if release[:64] != case['saved_next64']:
                        raise ValueError(f'first native 64 tokens differ from accepted trace: {image}')
                    if float((internal['first_logits'] - prefill.cpu()).abs().max()) > 2e-4:
                        raise ValueError('cached native first logits differ from exact prefill')
                    raw = list(result.raw_logprobs or ())
                    policy = list(result.policy_logprobs or ())
                else:
                    result, internal = _continue(qwen, batch, [*prefix, first], HORIZON - 1)
                    release = [first, *result.token_ids]
                    raw = [float(torch.log_softmax(prefill, 0)[first].item()), *(result.raw_logprobs or ())]
                    policy = raw[:1] + list(result.policy_logprobs or ())
                forwards += internal['generated_steps']
                if len(raw) != len(release) or len(policy) != len(release):
                    raise ValueError('token/logprob trace length changed')
                metrics = _analyze(prefix, release, qwen.tokenizer, case, result.stop_reason)
                cell = {'schema': 'first_illegal_coordinate_release.cell.v1', 'case': image, 'condition': condition,
                        'query_index': case['query_index'], 'prefix_sha256': case['prefix_sha256'],
                        'prompt_sha256': probe.digest_json(prompt), 'media_sha256': case['media_sha256'],
                        'first_token': first, 'first_token_is_forced': condition != 'native',
                        'first_logprob_unmodified': float(torch.log_softmax(prefill, 0)[first]),
                        'first_native_winner': int(prefill.argmax()), 'prefill_coordinate_full_logprob_max_error': prefill_max,
                        'cached_first_logit_max_error': float((internal['first_logits'] - prefill.cpu()).abs().max()) if condition == 'native' else None,
                        'token_ids': release, 'raw_logprobs': raw, 'policy_logprobs': policy,
                        'all_later_unmodified_argmax': True, 'stop_reason': result.stop_reason,
                        'metrics': metrics}
                probe.write_json(ROOT / f'cells/{image}-{condition}.json', cell)
        status = 'complete'
    finally:
        end = time.time()
        probe.write_json(ROOT / 'execution/terminal-v1.json', {'status': status, 'pid': os.getpid(),
                         'package_model_start': start, 'package_model_end': end,
                         'allocated_gpu_seconds': end - start, 'model_forwards': forwards})
    return ROOT / 'execution/terminal-v1.json'


def reduce(plan_path: Path, output: Path) -> Path:
    plan = json.loads(plan_path.read_text())
    rows = []
    for case in plan['cases']:
        for condition in plan['conditions']:
            path = ROOT / f'cells/{case["image_id"]}-{condition}.json'
            cell = json.loads(path.read_text())
            if (cell['case'] != case['image_id'] or cell['condition'] != condition or
                    cell['prefix_sha256'] != case['prefix_sha256'] or
                    len(cell['token_ids']) != len(cell['raw_logprobs'])):
                raise ValueError('missing or mismatched fixed cell')
            rows.append({'case': case['image_id'], 'condition': condition, 'cell': probe.binding(path),
                         'first_token': cell['first_token'], 'metrics': {k: v for k, v in cell['metrics'].items()
                                                                          if k not in ('release_text', 'complete_release_rows', 'parser_drops')},
                         'raw_rows': cell['metrics']['complete_release_rows'],
                         'parser_drops': cell['metrics']['parser_drops']})
    probe.write_json(output, {'schema': 'first_illegal_coordinate_release.reduction.v1',
                              'plan': probe.binding(plan_path), 'denominator': 6, 'observed': len(rows), 'rows': rows})
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'run', 'reduce'))
    parser.add_argument('--plan', type=Path, default=ROOT / 'selection/plan-v1.json')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--out', type=Path, default=ROOT / 'reduction/reduction-v1.json')
    args = parser.parse_args()
    result = prepare() if args.action == 'prepare' else run(args.plan, args.device) if args.action == 'run' else reduce(args.plan, args.out)
    print(result)


if __name__ == '__main__':
    main()
