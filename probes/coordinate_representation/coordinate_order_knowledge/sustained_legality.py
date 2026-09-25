"""Two fixed native continuations with a probe-local coordinate-slot logits mask."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
from transformers import LogitsProcessorList

from probes.coordinate_representation.coordinate_order_knowledge import first_illegal_release as release
from probes.coordinate_representation.coordinate_order_knowledge import probe
from src.inference.bound_requests import build_bound_native_requests
from src.qwen.generation import generate_continuations
from src.qwen.native import prepare_native_inputs


ROOT = Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-23-sustained-coordinate-legality')
PRIOR = release.ROOT
OLDER = release.PRIOR
COMMON_START = release.COMMON_START
PREVIOUS_GPU_SECONDS = 216.20124340057373
HORIZON = 384
STRUCTURE = {probe.REF_START, probe.REF_END, probe.BOX_START, probe.BOX_END, release.EOS}
SLOTS = {'x1', 'y1', 'x2', 'y2'}


class MalformedHistory(ValueError):
    pass


def scan_history(ids: list[int], coord_ids: list[int]) -> tuple[str, int | None, int | None]:
    """Parse only the compact row tokens; a partial final row is a state, not a repair."""
    bins = {token: i for i, token in enumerate(coord_ids)}
    stage = 'outside'
    x1 = y1 = None
    for index, token in enumerate(ids):
        if stage == 'outside':
            if token == probe.REF_START:
                stage = 'description'
            elif token == release.EOS:
                stage = 'stopped'
            else:
                raise MalformedHistory(f'unexpected token outside a row at {index}')
        elif stage == 'description':
            if token == probe.REF_END:
                stage = 'box_start'
            elif token in STRUCTURE or token in bins:
                raise MalformedHistory(f'broken description at {index}')
        elif stage == 'box_start':
            if token != probe.BOX_START:
                raise MalformedHistory(f'missing box start at {index}')
            stage = 'x1'
        elif stage in SLOTS:
            if token not in bins:
                raise MalformedHistory(f'non-coordinate token in {stage} at {index}')
            if stage == 'x1':
                x1, stage = bins[token], 'y1'
            elif stage == 'y1':
                y1, stage = bins[token], 'x2'
            elif stage == 'x2':
                stage = 'y2'
            else:
                stage = 'box_end'
        elif stage == 'box_end':
            if token != probe.BOX_END:
                raise MalformedHistory(f'missing box end at {index}')
            stage, x1, y1 = 'outside', None, None
        else:
            raise MalformedHistory(f'token after EOS at {index}')
    return stage, x1, y1


def legal_bins(stage: str, x1: int | None, y1: int | None) -> tuple[int, int] | None:
    if stage in ('x1', 'y1'):
        return 0, 998
    if stage == 'x2':
        if x1 is None or x1 == 999:
            raise MalformedHistory('x2 has no legal successor')
        return x1 + 1, 999
    if stage == 'y2':
        if y1 is None or y1 == 999:
            raise MalformedHistory('y2 has no legal successor')
        return y1 + 1, 999
    return None


class CoordinatePolicy:
    def __init__(self, prompt_width: int, coord_ids: list[int], *, enabled: bool = True):
        self.prompt_width = prompt_width
        self.coord_ids = coord_ids
        self.enabled = enabled
        self.records: list[dict] = []
        self.last_history: list[int] = []

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        if input_ids.shape[0] != 1 or scores.shape[0] != 1 or self.prompt_width >= input_ids.shape[1]:
            raise ValueError('policy requires one unpadded native history')
        self.last_history = [int(x) for x in input_ids[0, self.prompt_width:].tolist()]
        stage, x1, y1 = scan_history(self.last_history, self.coord_ids)
        legal = legal_bins(stage, x1, y1)
        info = {'stage': stage, 'x1': x1, 'y1': y1, 'legal_range': list(legal) if legal else None,
                'history_sha256': probe.digest_json(self.last_history), 'history_length': len(self.last_history)}
        self.records.append(info)
        if not self.enabled or legal is None:
            return scores
        allowed = torch.tensor(self.coord_ids[legal[0]:legal[1] + 1], device=scores.device)
        masked = torch.full_like(scores, -torch.inf)
        masked[:, allowed] = scores[:, allowed]
        return masked


def prepare() -> Path:
    old_path = PRIOR / 'selection/plan-v1.json'
    old = json.loads(old_path.read_text())
    if [c['image_id'] for c in old['cases']] != [885, 5586]:
        raise ValueError('accepted population changed')
    coord = old['coordinate_token_ids']
    if coord != list(range(probe.COORD0, probe.COORD0 + 1000)):
        raise ValueError('coordinate family ID/bin ordering changed')
    cases = []
    bindings = [probe.binding(old_path), probe.binding(Path(__file__)),
                probe.binding(PRIOR / 'lead-acceptance-v1.json'), probe.binding(probe.SOURCE / 'configs/resolved.json'),
                probe.binding(probe.SOURCE / 'image_plan.jsonl'), probe.binding(probe.DATA)]
    for case in old['cases']:
        prefix = case['prefix_token_ids']
        if probe.digest_json(prefix) != case['prefix_sha256'] or scan_history(prefix, coord)[0] != 'x2':
            raise ValueError('frozen first-illegal prefix does not end at x2')
        if legal_bins(*scan_history(prefix, coord)) != (1, 999):
            raise ValueError('first x2 threshold differs from 0')
        native_path = PRIOR / f'cells/{case["image_id"]}-native.json'
        native = json.loads(native_path.read_text())
        older_path = OLDER / f'cells/{case["image_id"]}-invalid_native-x2-{case["query_index"]}.json'
        older = json.loads(older_path.read_text())
        read = next(x['readout'] for x in older['conditions'] if x['axis'] == 'same' and x['requested_threshold'] == 0)
        if native['token_ids'][:8] != case['saved_next64'][:8] or native['first_native_winner'] != probe.COORD0:
            raise ValueError('accepted native eight-token replay or winner changed')
        bindings += [probe.binding(native_path), probe.binding(older_path), probe.binding(Path(case['image_plan']['image_path']))]
        cases.append({'image_id': case['image_id'], 'row_id': case['row_id'], 'query_index': case['query_index'],
                      'prefix_token_ids': prefix, 'prefix_sha256': case['prefix_sha256'],
                      'prompt_token_ids': case['prompt_token_ids'], 'media_sha256': case['media_sha256'],
                      'grid_thw': case['grid_thw'], 'image_plan': case['image_plan'],
                      'native_cell': probe.binding(native_path), 'source_readout_cell': probe.binding(older_path),
                      'source_coordinate_full_logprobs': read['coordinate_logprobs_full_vocabulary'],
                      'native_first8': native['token_ids'][:8]})
    plan = {'schema': 'sustained_coordinate_legality.plan.v1', 'cases': cases,
            'coordinate_token_ids': coord, 'bindings': bindings, 'source_bindings': old['source_bindings'],
            'runtime': old['source_runtime'], 'conditions': ['native_reused', 'sustained_joint_type_order'],
            'policy': {'x1_y1': [0, 998], 'x2': 'bin > same-row x1', 'y2': 'bin > same-row y1',
                       'outside_coordinate_slots': 'identity', 'temperature': 0, 'top_p': 1,
                       'top_k': 0, 'repetition_penalty': 1, 'use_model_defaults': False},
            'horizon': HORIZON, 'shared_start': COMMON_START, 'previous_forwards': 2339,
            'previous_gpu_seconds': PREVIOUS_GPU_SECONDS,
            'package_limits': {'wall_seconds': 3600, 'gpu_seconds': 3600, 'forwards': 1024,
                               'artifact_bytes': 1024**3, 'admit_until_seconds': 3300}}
    path = ROOT / 'selection/plan-v1.json'
    probe.write_json(path, plan)
    return path


def _generate(qwen, batch, prefix: list[int], policy: CoordinatePolicy | None, budget: int):
    model = qwen.model
    original = model.generate
    captured = []

    def wrapper(**kwargs):
        if policy is not None:
            kwargs['logits_processor'] = LogitsProcessorList([policy])
        output = original(**kwargs)
        width = kwargs['input_ids'].shape[1]
        emitted = [int(x) for x in output.sequences[0, width:].tolist()]
        raw = tuple(output.logits)
        processed = tuple(output.scores)
        if len(raw) != len(processed) or len(raw) != len(emitted):
            raise ValueError('generated token/logit alignment changed')
        if policy is not None and len(policy.records) != len(emitted):
            raise ValueError('policy was not invoked exactly once per emitted token')
        steps = []
        for i, (a, b, token) in enumerate(zip(raw, processed, emitted, strict=True)):
            a, b = a[0].float(), b[0].float()
            spec = {'stage': 'off', 'legal_range': None} if policy is None else policy.records[i]
            legal = spec['legal_range'] if policy is not None and policy.enabled else None
            if legal is None:
                if float((a - b).abs().max()) > 2e-4:
                    raise ValueError('outside-slot logits changed')
            else:
                allowed = torch.tensor(policy.coord_ids[legal[0]:legal[1] + 1], device=a.device)
                if not torch.equal(b[allowed], a[allowed]) or int(torch.isfinite(b).sum()) != len(allowed):
                    raise ValueError('coordinate mask differs from exact legal family')
                if token not in policy.coord_ids[legal[0]:legal[1] + 1]:
                    raise ValueError('emitted illegal coordinate')
            if token != int(b.argmax()):
                raise ValueError('emitted token is not constrained greedy argmax')
            steps.append({'index': i, 'emitted_token': token, 'raw_argmax': int(a.argmax()),
                          'policy_argmax': int(b.argmax()), 'raw_logprob': float(torch.log_softmax(a, 0)[token]),
                          'policy_logprob': float(torch.log_softmax(b, 0)[token]),
                          'stage': spec['stage'], 'x1': spec.get('x1'), 'y1': spec.get('y1'),
                          'legal_range': legal, 'history_sha256': spec.get('history_sha256')})
        captured.append({'steps': steps, 'first_raw_logits': raw[0][0].float().cpu(),
                         'first_policy_logits': processed[0][0].float().cpu()})
        return output

    model.generate = wrapper
    try:
        result = generate_continuations(model, batch, extensions=[prefix], budgets=[budget],
                                        eos_token_id=release.EOS, pad_token_id=qwen.tokenizer.pad_token_id,
                                        policy=release.POLICY, trace='raw_and_policy', seed=None)[0]
    finally:
        model.generate = original
    if len(captured) != 1 or len(captured[0]['steps']) != len(result.token_ids):
        raise ValueError('one unpadded cached generation was not returned')
    return result, captured[0]


def run(plan_path: Path, device: str = 'cuda:0') -> Path:
    from src.artifacts.source_provenance import preserve_source

    plan = json.loads(plan_path.read_text())
    if plan['schema'] != 'sustained_coordinate_legality.plan.v1' or len(plan['cases']) != 2:
        raise ValueError('frozen plan changed')
    for b in [*plan['bindings'], *(x for x in plan['source_bindings'] if Path(x['path']).name != 'probe.py')]:
        if probe.binding(Path(b['path']))['sha256'] != b['sha256']:
            raise ValueError(f'input/source binding changed: {b["path"]}')
    captures = []
    for rel in ('probes/coordinate_representation/coordinate_order_knowledge/sustained_legality.py',
                'probes/coordinate_representation/coordinate_order_knowledge/probe.py',
                'probes/coordinate_representation/coordinate_order_knowledge/first_illegal_release.py',
                'src/qwen/native.py', 'src/qwen/generation.py', 'src/inference/hf_backend.py',
                'src/inference/parsing.py', 'src/inference/bound_requests.py'):
        current = Path.cwd() / rel
        saved = preserve_source(current, run_root=ROOT, relative_name=rel)
        captures.append({'current': probe.binding(current), 'capture': probe.binding(saved)})
    start = time.time()
    probe.write_json(ROOT / 'execution/launch-v1.json', {'plan': probe.binding(plan_path),
                     'captures': captures, 'device': device, 'pid': os.getpid(),
                     'argv': list(__import__('sys').argv), 'package_model_start': start, 'shared_model_start': COMMON_START})
    config = json.loads((probe.SOURCE / 'configs/resolved.json').read_text())['config']
    forwards = 0
    status = 'failed'
    try:
        qwen = probe._load_source(config, device)
        data = {x['image_id']: x for x in probe.jsonl(probe.DATA)}
        for case in plan['cases']:
            if time.time() - start > 3300 or time.time() - COMMON_START > 5.75 * 3600 or forwards + 400 > 1024:
                raise TimeoutError('new model admission reserve reached')
            image = case['image_id']
            plan_row = case['image_plan']
            row = {'row_id': case['row_id'], 'row_index': 0, 'input_record': data[image],
                   'image_path': plan_row['image_path'], 'image_width': plan_row['decoded_width'],
                   'image_height': plan_row['decoded_height'], 'image_plan': plan_row}
            requests, _ = build_bound_native_requests(qwen, config, [row])
            batch = prepare_native_inputs(qwen.processor, requests, device=device, record_media_identity=True)
            prompt = list(batch.prompt_token_ids[0])
            if (prompt != case['prompt_token_ids'] or list(batch.media_sha256 or ()) != [case['media_sha256']]
                    or list(batch.image_grids[0]) != case['grid_thw']):
                raise ValueError('accepted prompt/media/grid changed')
            prefix = case['prefix_token_ids']
            if scan_history(prefix, plan['coordinate_token_ids'])[0] != 'x2':
                raise ValueError('initial query slot changed')
            prefill = probe._native_logits(qwen, batch.inputs, [*prompt, *prefix], [0])
            forwards += 1
            read = probe.readout(prefill, plan['coordinate_token_ids'], 0)['coordinate_logprobs_full_vocabulary']
            prefill_max = max(abs(a - b) for a, b in zip(read, case['source_coordinate_full_logprobs'], strict=True))
            if prefill_max > 2e-4 or int(prefill.argmax()) != probe.COORD0:
                raise ValueError('accepted source prefill changed')
            off, off_trace = _generate(qwen, batch, prefix, None, 8)
            forwards += len(off.token_ids)
            if list(off.token_ids) != case['native_first8'] or float((off_trace['first_raw_logits'] - prefill.cpu()).abs().max()) > 2e-4:
                raise ValueError('OFF cached eight-token source replay changed')
            probe.write_json(ROOT / f'qualification/{image}-off8.json', {'case': image,
                             'tokens': list(off.token_ids), 'raw_step_logprobs': [s['raw_logprob'] for s in off_trace['steps']],
                             'prefill_coordinate_max': prefill_max,
                             'cached_first_full_logit_max': float((off_trace['first_raw_logits'] - prefill.cpu()).abs().max())})
            policy = CoordinatePolicy(len(prompt), plan['coordinate_token_ids'])
            try:
                result, trace = _generate(qwen, batch, prefix, policy, HORIZON)
                forwards += len(result.token_ids)
                ids = list(result.token_ids)
                # A final partial row is permitted; a malformed token sequence is not.
                final_stage = scan_history([*prefix, *ids], plan['coordinate_token_ids'])[0]
                if final_stage == 'stopped' and ids[-1] != release.EOS:
                    raise ValueError('EOS appeared before the terminal generated token')
                rows = release._rows([*prefix, *ids], qwen.tokenizer)
                for r in rows:
                    if r['end'] >= len(prefix) and (r['equality_x'] or r['equality_y'] or r['reversal_x'] or r['reversal_y']):
                        raise ValueError('policy emitted invalid complete box before parser drops')
                metrics = release._analyze(prefix, ids, qwen.tokenizer, case, result.stop_reason)
                cell = {'schema':'sustained_coordinate_legality.cell.v1','status':'complete','case':image,
                        'prefix_sha256':case['prefix_sha256'],'prompt_sha256':probe.digest_json(prompt),
                        'media_sha256':case['media_sha256'],'first_native_winner':int(prefill.argmax()),
                        'prefill_coordinate_max':prefill_max,'token_ids':ids,'steps':trace['steps'],
                        'stop_reason':result.stop_reason,'final_partial_stage':final_stage,'metrics':metrics}
                probe.write_json(ROOT / f'cells/{image}-sustained.json', cell)
            except MalformedHistory as exc:
                partial = policy.last_history[len(prefix):]
                probe.write_json(ROOT / f'cells/{image}-hold.json', {'case':image,'status':'HOLD',
                         'reason':str(exc),'partial_generated_token_ids':partial,
                         'policy_records':policy.records,'prefix_sha256':case['prefix_sha256']})
                forwards += len(policy.records)
                raise
        status = 'complete'
    finally:
        end = time.time()
        probe.write_json(ROOT / 'execution/terminal-v1.json', {'status':status,'pid':os.getpid(),
                         'package_model_start':start,'package_model_end':end,
                         'allocated_gpu_seconds':end-start,'model_forwards':forwards})
    return ROOT / 'execution/terminal-v1.json'


def reduce(plan_path: Path, output: Path) -> Path:
    plan = json.loads(plan_path.read_text())
    rows = []
    for case in plan['cases']:
        native_path = Path(case['native_cell']['path'])
        if probe.binding(native_path)['sha256'] != case['native_cell']['sha256']:
            raise ValueError('accepted native cell mutated')
        native = json.loads(native_path.read_text())
        path = ROOT / f'cells/{case["image_id"]}-sustained.json'
        cell = json.loads(path.read_text())
        if cell['status'] != 'complete' or cell['case'] != case['image_id'] or cell['prefix_sha256'] != case['prefix_sha256']:
            raise ValueError('new cell identity or status changed')
        rows.append({'case':case['image_id'],'native':{'binding':case['native_cell'],'metrics':native['metrics']},
                     'sustained':{'binding':probe.binding(path),'metrics':cell['metrics']},
                     'new_step_count':len(cell['steps'])})
    probe.write_json(output, {'schema':'sustained_coordinate_legality.reduction.v1','planned_new_cells':2,
                              'complete_new_cells':len(rows),'reused_native_cells':2,'rows':rows})
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare','run','reduce'))
    parser.add_argument('--plan',type=Path,default=ROOT/'selection/plan-v1.json')
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--out',type=Path,default=ROOT/'reduction/reduction-v1.json')
    args = parser.parse_args()
    result = prepare() if args.action == 'prepare' else run(args.plan,args.device) if args.action == 'run' else reduce(args.plan,args.out)
    print(result)


if __name__ == '__main__':
    main()
