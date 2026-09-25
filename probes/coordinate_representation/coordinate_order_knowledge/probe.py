"""Prepare, execute and reduce the frozen coordinate-token order diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoProcessor

from src.inference.bound_requests import build_bound_native_requests
from src.qwen.native import exact_history_inputs, prepare_native_inputs
from probes.coordinate_representation.coordinate_codebook_alignment import evaluation


ROOT = Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-23-coordinate-order-knowledge')
SOURCE = Path('/data/CoordExp/outputs/archive/research/qwen3-vl-dense-enumeration/2026-09-23-codebook-paired2048-invalid-mean-hinge/inference/paired2048-source-val200')
DATA = Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-23-codebook-paired2048-preparation/data/val200-xyxy-v2.coord.jsonl')
REUSE = Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-23-codebook-paired2048-ordergate/qualification/source-reuse-v1.json')
INVALID = (632, 885, 5586, 7281, 18380)
COORD0 = 151670
BOX_START, BOX_END, REF_START, REF_END = 151648, 151649, 151646, 151647
SOURCE_COMMIT = '6b6883529c6d6996f1826231406dd47e96a2abcf'
MAX_FORWARDS = 4096
MAX_WALL = 7200
MAX_GPU_SECONDS = 14400
ADMIT_UNTIL = 6900
MAX_BYTES = 2 * 1024**3


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_json(value: object) -> str:
    return digest_bytes(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode())


def binding(path: Path) -> dict:
    return {'path': str(path.resolve()), 'sha256': digest_bytes(path.read_bytes()), 'size_bytes': path.stat().st_size}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')) + '\n').encode()
    with path.open('xb') as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def row_id(image_id: int) -> str:
    return f'coco2017_val_{image_id:012d}'


def boxes(tokens: list[dict]) -> list[dict]:
    """Only complete literal boxes; malformed states cannot become negatives."""
    ids = [int(t['token_id']) for t in tokens]
    result = []
    for j, token in enumerate(ids):
        if token != BOX_START or j + 5 >= len(ids):
            continue
        if ids[j + 5] != BOX_END or any(not COORD0 <= x < COORD0 + 1000 for x in ids[j + 1:j + 5]):
            continue
        refs = [k for k in range(j) if ids[k] == REF_START]
        if not refs or REF_END not in ids[refs[-1] + 1:j]:
            continue
        ref = refs[-1]
        end = ids.index(REF_END, ref + 1, j)
        if end != j - 1:
            continue
        bins = [x - COORD0 for x in ids[j + 1:j + 5]]
        result.append({'box_start': j, 'coord_bins': bins,
                       'description': ''.join(str(t['token_text']) for t in tokens[ref + 1:end]),
                       'invalid_x': bins[0] >= bins[2], 'invalid_y': bins[1] >= bins[3]})
    return result


def nearest_valid(candidate: list[dict], target: int) -> dict | None:
    valid = [b for b in candidate if not b['invalid_x'] and not b['invalid_y']]
    return min(valid, key=lambda b: (abs(b['box_start'] - target), b['box_start'])) if valid else None


def thresholds(a: int, v: int) -> list[dict]:
    if not 0 <= a <= 999 or not 0 <= v <= 999:
        raise ValueError('coordinate bin outside 0..999')
    values = {a: {'identity'}}
    for delta in (-4, -1, 1, 4):
        values.setdefault(min(998, max(0, a + delta)), set()).add(f'offset_{delta:+d}')
    if 1 <= v <= 998:
        values.setdefault(v - 1, set()).add('candidate_legal_side')
        values.setdefault(v, set()).add('candidate_illegal_side')
    return [{'threshold': x, 'reasons': sorted(why)} for x, why in sorted(values.items())]


def case_queries(tokens: list[dict], box: dict, kind: str) -> list[dict]:
    j = box['box_start']
    ids = [int(t['token_id']) for t in tokens]
    result = []
    for role, q, preceding, wrong in (('x2', j + 3, j + 1, j + 2), ('y2', j + 4, j + 2, j + 1)):
        if not (0 <= preceding < q < len(ids)) or ids[preceding] != COORD0 + box['coord_bins'][0 if role == 'x2' else 1]:
            raise ValueError('swapped role or shifted causal prefix')
        if role == 'y2' and ids[q - 1] != COORD0 + box['coord_bins'][2]:
            raise ValueError('y2 must condition on this box x2')
        a = ids[preceding] - COORD0
        v = ids[q] - COORD0
        if not (0 <= a <= 999 and 0 <= v <= 999 and 0 <= ids[wrong] - COORD0 <= 999):
            raise ValueError('query or predecessor is not a coordinate token')
        result.append({'kind': kind, 'role': role, 'query_index': q, 'preceding_index': preceding,
                       'wrong_axis_index': wrong, 'original_threshold': a, 'original_wrong_axis': ids[wrong] - COORD0,
                       'observed_successor': v, 'original_role_illegal': v <= a,
                       'description': box['description'], 'box_bins': box['coord_bins'],
                       'thresholds': thresholds(a, v), 'prefix_token_ids': ids[:q],
                       'prefix_sha256': digest_json(ids[:q]), 'observed_token_id': ids[q]})
    return result


def validate_query(query: dict) -> None:
    q = query['query_index']
    prefix = query['prefix_token_ids']
    full = query['history_token_ids']
    if prefix != full[:q] or full[q] != query['observed_token_id']:
        raise ValueError('future suffix or shifted causal target entered the query')
    if query['role'] == 'x2':
        expected = (q - 2, q - 1)
        schema = prefix[q - 3]
    elif query['role'] == 'y2':
        expected = (q - 2, q - 3)
        schema = prefix[q - 4]
        if prefix[q - 1] != COORD0 + query['box_bins'][2]:
            raise ValueError('y2 history does not include this row x2')
    else:
        raise ValueError('only actual x2/y2 queries are admitted')
    if schema != BOX_START or (query['preceding_index'], query['wrong_axis_index']) != expected:
        raise ValueError('swapped role or shifted predecessor token index')
    if prefix[expected[0]] != COORD0 + query['original_threshold'] or prefix[expected[1]] != COORD0 + query['original_wrong_axis']:
        raise ValueError('same-row causal threshold differs from frozen metadata')


def iou(a: list[int], b: list[int]) -> float:
    w = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    both = w * h
    aa = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    bb = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return both / (aa + bb - both) if aa + bb > both else 0.0


def textual_pairs() -> list[list[int]]:
    adjacent = [(x, x + 1) for x in (0, 1, 8, 9, 10, 18, 19, 20, 98, 99, 100, 198, 199, 200, 498, 499, 500, 898, 998)]
    boundaries = [(x - 1, x + 1) for x in (10, 20, 30, 40, 50, 90, 100, 110, 200, 300, 400, 500, 600, 700, 800, 900, 990, 998)]
    broad = [(x, y) for x, y in zip(range(0, 405, 15), range(999, 594, -15))]
    pairs = list(dict.fromkeys(adjacent + boundaries + broad))[:64]
    if len(pairs) != 64 or any(not 0 <= a < b <= 999 for a, b in pairs):
        raise AssertionError('textual pair plan must contain 64 distinct legal ordered pairs')
    return [list(pair) for pair in pairs]


def _source_bindings(config: dict) -> list[dict]:
    paths = [SOURCE / n for n in ('configs/resolved.json', 'run_manifest.json', 'gt_vs_pred.jsonl',
              'pred_token_trace.jsonl', 'parse_diagnostics.jsonl', 'image_plan.jsonl')]
    paths += [DATA, REUSE, Path(__file__), Path(config['model']['base_model']) / 'config.json']
    for key in ('adapter', 'embedding_delta'):
        paths += sorted(p for p in Path(config[key]['path']).rglob('*') if p.is_file())
    return [binding(p) for p in paths]


def prepare() -> Path:
    import subprocess
    from src.inference.bound_requests import build_bound_native_requests

    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() != SOURCE_COMMIT:
        raise ValueError('closed source HEAD changed')
    resolved = json.loads((SOURCE / 'configs/resolved.json').read_text())['config']
    if resolved['backend']['hf']['adapter_runtime'] != 'live_promoted' or resolved['model']['dtype'] != 'fp32' or resolved['backend']['hf']['attn_implementation'] != 'sdpa':
        raise ValueError('accepted source runtime differs')
    accepted = {Path(b['path']).resolve(): b for b in json.loads(REUSE.read_text())['bindings']}
    for path, saved in accepted.items():
        if not path.is_file() or digest_bytes(path.read_bytes()) != saved['sha256']:
            raise ValueError(f'accepted source reuse binding changed: {path}')
    source_rows = {x['row_id']: x for x in jsonl(SOURCE / 'gt_vs_pred.jsonl')}
    plans = {x['row_id']: x for x in jsonl(SOURCE / 'image_plan.jsonl')}
    dataset = {row_id(x['image_id']): (i, x) for i, x in enumerate(jsonl(DATA))}
    generated = defaultdict(list)
    for x in jsonl(SOURCE / 'pred_token_trace.jsonl'):
        if x['trace_type'] == 'generated_token' and not x['is_pad']:
            generated[x['row_id']].append(x)
    for rid, tokens in generated.items():
        tokens.sort(key=lambda t: t['generated_step_index'])
        if [x['generated_step_index'] for x in tokens] != list(range(len(tokens))):
            raise ValueError(f'noncontiguous generated trace: {rid}')
    found = {rid: boxes(tokens) for rid, tokens in generated.items()}
    invalid_info = {}
    for image_id in INVALID:
        rid = row_id(image_id)
        first = next((b for b in found[rid] if b['invalid_x'] or b['invalid_y']), None)
        if first is None:
            raise ValueError(f'frozen invalid image has no complete illegal box: {rid}')
        invalid_info[rid] = first
    healthy = {rid for rid, row in source_rows.items() if rid not in invalid_info and
               row['dropped_prediction_count'] == 0 and row['decode_stop_reason'] == 'im_end' and found[rid]}
    selected_controls = []
    matches = {}
    for rid in [row_id(i) for i in INVALID] + [row_id(632), row_id(18380)]:
        target = invalid_info[rid]['box_start']
        n = len(source_rows[rid]['gt'])
        ranked = []
        for other in healthy - set(selected_controls):
            nearest = nearest_valid(found[other], target)
            if nearest is None:
                continue
            count = len(source_rows[other]['gt'])
            distance = abs(count - n) / max(1, n) + abs(nearest['box_start'] - target) / max(1, target)
            ranked.append((distance, int(other[-12:]), other, nearest))
        if not ranked:
            raise ValueError('not enough healthy control support')
        _, _, chosen, box = min(ranked)
        selected_controls.append(chosen)
        matches[chosen] = {'matched_invalid': rid, 'annotation_count': len(source_rows[chosen]['gt']),
                           'reference_annotation_count': n, 'box_start': box['box_start'],
                           'reference_box_start': target, 'distance': min(ranked)[0]}
    if len(selected_controls) != 7:
        raise AssertionError('control denominator changed')
    processor = AutoProcessor.from_pretrained(resolved['model']['base_model'], local_files_only=True)
    processor.image_processor.do_resize = False
    qwen = SimpleNamespace(processor=processor, tokenizer=processor.tokenizer)
    coord_ids = [processor.tokenizer.convert_tokens_to_ids(f'<|coord_{i}|>') for i in range(1000)]
    if coord_ids != list(range(COORD0, COORD0 + 1000)):
        raise ValueError('coordinate token order differs from accepted source')
    cases = []
    for rid in [*(row_id(i) for i in INVALID), *selected_controls]:
        original = source_rows[rid]
        index, record = dataset[rid]
        plan = plans[rid]
        case = {'row_id': rid, 'row_index': index, 'input_record': record,
                'image_path': plan['image_path'], 'image_width': plan['decoded_width'],
                'image_height': plan['decoded_height'], 'image_plan': plan}
        requests, _ = build_bound_native_requests(qwen, resolved, [case])
        batch = prepare_native_inputs(processor, requests, device='cpu', record_media_identity=True)
        prompt_ids = list(batch.prompt_token_ids[0])
        if tuple(batch.image_grids[0]) != tuple(plan['observed_image_grid_thw']) or list(batch.media_sha256 or ()) != [plan['executed_media_sha256']]:
            raise ValueError(f'processor image/grid identity changed: {rid}')
        target = invalid_info[rid] if rid in invalid_info else nearest_valid(found[rid], matches[rid]['reference_box_start'])
        if target is None:
            raise ValueError(f'healthy control has no valid box: {rid}')
        first_valid = [b for b in found[rid] if b['box_start'] < target['box_start'] and not b['invalid_x'] and not b['invalid_y']]
        picked = [('invalid_native' if rid in invalid_info else 'healthy_native', target)]
        if first_valid:
            picked.append(('prior_valid_native', first_valid[-1]))
        queries = [q for kind, box in picked for q in case_queries(generated[rid], box, kind)]
        for q in queries:
            q['history_token_ids'] = [int(t['token_id']) for t in generated[rid]]
            q['saved_selected_logprob'] = generated[rid][q['query_index']]['logprob']
            validate_query(q)
        teacher = []
        target_ids = evaluation._target_ids(qwen, case, resolved, DATA)
        target_tokens = [{'token_id': token, 'token_text': processor.tokenizer.decode([token])} for token in target_ids]
        gt_boxes = boxes(target_tokens)
        for kind, box in picked:
            if box['invalid_x'] or box['invalid_y']:
                teacher.append({'kind': kind, 'status': 'HOLD', 'reason': 'invalid native geometry has no IoU owner'})
                continue
            candidates = [i for i, obj in enumerate(record['objects']) if obj['desc'] == box['description'] and
                          iou(box['coord_bins'], [int(x[len('<|coord_'):-2]) for x in obj['bbox_2d']]) >= 0.5]
            if len(candidates) != 1:
                teacher.append({'kind': kind, 'status': 'HOLD', 'reason': 'no unique same-class IoU50 annotation', 'candidate_count': len(candidates)})
                continue
            obj = record['objects'][candidates[0]]
            gt_bins = [int(x[len('<|coord_'):-2]) for x in obj['bbox_2d']]
            target_matches = [b for b in gt_boxes if b['coord_bins'] == gt_bins and b['description'] == obj['desc']]
            if len(target_matches) != 1:
                teacher.append({'kind': kind, 'status': 'HOLD', 'reason': 'rendered annotation row is not unique'})
                continue
            for q in case_queries(target_tokens, target_matches[0], 'teacher'):
                q['history_token_ids'] = target_ids
                q['source_native_kind'] = kind
                q['annotation_owner_id'] = str(obj['coco_ann_id'])
                validate_query(q)
                teacher.append(q)
        cases.append({'image_id': int(rid[-12:]), 'row_id': rid, 'status': 'source_invalid' if rid in invalid_info else 'healthy_control',
                      'match': matches.get(rid), 'annotation_count': len(record['objects']),
                      'source_stop': original['decode_stop_reason'], 'source_generated_tokens': len(generated[rid]),
                      'image_content_sha256': plan['image_content_sha256'], 'media_sha256': plan['executed_media_sha256'],
                      'grid_thw': plan['observed_image_grid_thw'], 'prompt_token_ids': prompt_ids,
                      'prompt_sha256': digest_json(prompt_ids), 'queries': queries,
                      'teacher': teacher, 'native_boxes': picked})
    source_bindings = _source_bindings(resolved)
    plan = {'schema': 'coordinate_order_knowledge.plan.v1', 'source_commit': SOURCE_COMMIT,
            'source_runtime': {'dtype': 'fp32', 'attention': 'sdpa', 'adapter_runtime': 'live_promoted'},
            'source_bindings': source_bindings, 'coordinate_token_ids': coord_ids,
            'invalid_image_ids': list(INVALID), 'control_image_ids': [int(x[-12:]) for x in selected_controls],
            'selection_rule': 'greedy unmatched minimum |annotation-count delta|/reference count + |nearest valid box-start delta|/reference box-start; five originals then extras for 632,18380',
            'cases': cases, 'textual_pairs': textual_pairs(),
            'textual_contract': {'orders': ['ascending', 'descending'], 'answer_options': ['A_yes', 'B_yes'],
                                 'interfaces': ['coord_token', 'ordinary_decimal'], 'answer_labels': ['A', 'B']},
            'limits': {'model_wall_seconds': MAX_WALL, 'allocated_gpu_seconds': MAX_GPU_SECONDS,
                       'model_forwards': MAX_FORWARDS, 'artifact_bytes': MAX_BYTES, 'admission_cutoff_seconds': ADMIT_UNTIL}}
    path = ROOT / 'selection' / 'plan-v1.json'
    write_json(path, plan)
    return path


def _load_source(config: dict, device: str):
    from src.inference.backend import BackendLaunch
    from src.inference.hf_backend import _load_hf_components

    launch = BackendLaunch(
        backend='hf', model_path=config['model']['base_model'], model_dtype='fp32', batch_size=1,
        generation_config_fingerprint=digest_json(config['generation']),
        backend_options={'hf': {'attn_implementation': 'sdpa', 'patch_embed_linearization': 'enabled',
                                'adapter_runtime': 'live_promoted'}},
        adapter=config['adapter'], embedding_delta=config['embedding_delta'],
    )
    loaded = _load_hf_components(launch)
    qwen = loaded.qwen
    qwen.model.to(torch.device(device)).eval()
    qwen.processor.image_processor.do_resize = False
    if getattr(qwen.model, 'coordinate_codebook', None) is not None:
        raise ValueError('source unexpectedly contains a codebook')
    return qwen


def readout(logits: torch.Tensor, coordinate_ids: list[int], threshold: int) -> dict:
    if logits.ndim != 1 or not torch.isfinite(logits).all() or not 0 <= threshold <= 999:
        raise ValueError('invalid full-vocabulary logits or threshold')
    z = logits.float()
    c = z[torch.tensor(coordinate_ids, device=z.device)]
    full = torch.logsumexp(z, 0)
    family = torch.logsumexp(c, 0)
    conditional = c - family
    family_mass = (family - full).exp()
    illegal = conditional[:threshold + 1].exp().sum()
    return {'coordinate_logprobs_conditional': conditional.cpu().tolist(),
            'coordinate_logprobs_full_vocabulary': (c - full).cpu().tolist(),
            'full_vocabulary_winner': int(z.argmax().item()),
            'family_winner_bin': int(c.argmax().item()),
            'coordinate_family_mass': float(family_mass),
            'illegal_family_mass': float(illegal),
            'illegal_full_vocabulary_mass': float(illegal * family_mass),
            'equality_bin_probability': float(conditional[threshold].exp()),
            'full_log_normalizer': float(full)}


def _native_logits(qwen, native_inputs: dict, history: list[int], forwards: list[int]) -> torch.Tensor:
    if forwards[0] >= MAX_FORWARDS:
        raise RuntimeError('model-forward cap reached')
    kwargs = exact_history_inputs(qwen.model, native_inputs, [history], pad_token_id=0, logits_to_keep=1)
    observed = kwargs['input_ids'][0][kwargs['attention_mask'][0].bool()].tolist()
    if observed != history or kwargs['logits_to_keep'] != 1:
        raise ValueError('exact history or causal logit position changed')
    with torch.no_grad():
        logits = qwen.model(**kwargs).logits
    forwards[0] += 1
    if logits.ndim != 3 or logits.shape[0:2] != (1, 1):
        raise ValueError('native replay returned wrong causal shape')
    return logits[0, -1].float()


def _text_prompt(tokenizer, a: int, b: int, *, representation: str, yes_label: str) -> list[int]:
    if representation == 'coord_token':
        first, second = f'<|coord_{a}|>', f'<|coord_{b}|>'
    elif representation == 'ordinary_decimal':
        first, second = f'{a:03d}', f'{b:03d}'
    else:
        raise ValueError('unknown textual interface')
    no_label = 'B' if yes_label == 'A' else 'A'
    content = (f'Which value is greater? First: {first}. Second: {second}. '
               f'Answer {yes_label} if first is greater, or {no_label} if second is greater. '
               'Reply with exactly one letter.')
    text = tokenizer.apply_chat_template([{'role': 'user', 'content': content}], tokenize=False, add_generation_prompt=True)
    ids = tokenizer.encode(text, add_special_tokens=False)
    coords = [i for i in ids if COORD0 <= i < COORD0 + 1000]
    if representation == 'coord_token' and coords != [COORD0 + a, COORD0 + b]:
        raise ValueError('coordinate interface did not contain exactly the frozen coordinate tokens')
    if representation == 'ordinary_decimal' and coords:
        raise ValueError('decimal interface unexpectedly contains coordinate tokens')
    return ids


def _guard(start: float, forwards: int, output_dir: Path) -> None:
    elapsed = time.time() - start
    if elapsed >= ADMIT_UNTIL or elapsed >= MAX_WALL or elapsed >= MAX_GPU_SECONDS or forwards >= MAX_FORWARDS:
        raise TimeoutError('frozen wall/GPU/forward admission limit reached')
    if sum(p.stat().st_size for p in output_dir.rglob('*') if p.is_file()) >= MAX_BYTES:
        raise RuntimeError('artifact-byte limit reached')


def run(plan_path: Path, device: str = 'cuda:0') -> Path:
    plan = json.loads(plan_path.read_text())
    if plan['schema'] != 'coordinate_order_knowledge.plan.v1' or len(plan['cases']) != 12 or len(plan['textual_pairs']) != 64:
        raise ValueError('frozen panel or pair denominator changed')
    # The selection bytes remain frozen; the launch binds the final executed code.
    for saved in plan['source_bindings']:
        path = Path(saved['path'])
        if path.resolve() == Path(__file__).resolve():
            continue
        if binding(path)['sha256'] != saved['sha256']:
            raise ValueError(f'frozen input binding changed: {path}')
    config = json.loads((SOURCE / 'configs/resolved.json').read_text())['config']
    from src.artifacts.source_provenance import preserve_source
    captured = []
    for relative in ('probes/coordinate_representation/coordinate_order_knowledge/probe.py',
                     'src/qwen/native.py', 'src/inference/bound_requests.py',
                     'src/inference/hf_backend.py', 'src/adapters/dora.py'):
        source = Path.cwd() / relative
        preserved = preserve_source(source, run_root=ROOT, relative_name=relative)
        captured.append({'current': binding(source), 'capture': binding(preserved)})
    launch_path = ROOT / 'execution' / 'launch-v1.json'
    write_json(launch_path, {'schema': 'coordinate_order_knowledge.launch.v1', 'plan': binding(plan_path),
                             'source_config': binding(SOURCE / 'configs/resolved.json'),
                             'source_captures': captured, 'device': device, 'pid': os.getpid(),
                             'argv': list(__import__('sys').argv), 'model_wall_start': time.time()})
    start = json.loads(launch_path.read_text())['model_wall_start']
    cells = ROOT / 'cells'
    cells.mkdir(parents=True, exist_ok=True)
    forwards = [0]
    status = 'running'
    error = None
    try:
        qwen = _load_source(config, device)
        coord = [int(x) for x in qwen.token_identity.coordinate_token_ids]
        if coord != plan['coordinate_token_ids']:
            raise ValueError('loaded source coordinate token IDs differ from frozen plan')
        first_baseline = None
        for case in plan['cases']:
            _guard(start, forwards[0], ROOT)
            row = {'row_id': case['row_id'], 'row_index': next(i for i, x in enumerate(jsonl(DATA)) if x['image_id'] == case['image_id']),
                   'input_record': next(x for x in jsonl(DATA) if x['image_id'] == case['image_id']),
                   'image_path': next(x['image_path'] for x in jsonl(SOURCE / 'image_plan.jsonl') if x['row_id'] == case['row_id']),
                   'image_width': next(x['decoded_width'] for x in jsonl(SOURCE / 'image_plan.jsonl') if x['row_id'] == case['row_id']),
                   'image_height': next(x['decoded_height'] for x in jsonl(SOURCE / 'image_plan.jsonl') if x['row_id'] == case['row_id']),
                   'image_plan': next(x for x in jsonl(SOURCE / 'image_plan.jsonl') if x['row_id'] == case['row_id'])}
            requests, _ = build_bound_native_requests(qwen, config, [row])
            batch = prepare_native_inputs(qwen.processor, requests, device=device, record_media_identity=True)
            prompt = list(batch.prompt_token_ids[0])
            if prompt != case['prompt_token_ids'] or list(batch.media_sha256 or ()) != [case['media_sha256']] or list(batch.image_grids[0]) != case['grid_thw']:
                raise ValueError(f'prepared prompt/media/grid identity changed: {case["row_id"]}')
            for query in [*case['queries'], *(x for x in case['teacher'] if x.get('kind') == 'teacher')]:
                qname = f'{case["image_id"]}-{query["kind"]}-{query["role"]}-{query["query_index"]}'
                qpath = cells / f'{qname}.json'
                if qpath.exists():
                    raise FileExistsError(f'planned cell already exists: {qpath}')
                validate_query(query)
                original = list(query['prefix_token_ids'])
                conditions = []
                baseline_logits = None
                ordered_thresholds = sorted(query['thresholds'], key=lambda item: (item['threshold'] != query['original_threshold'], item['threshold']))
                for spec in ordered_thresholds:
                    for axis in ('same', 'wrong'):
                        if axis == 'wrong' and spec['threshold'] == query['original_threshold']:
                            continue
                        _guard(start, forwards[0], ROOT)
                        history = [*prompt, *original]
                        changed = spec['threshold']
                        if axis == 'same':
                            history[len(prompt) + query['preceding_index']] = COORD0 + changed
                            observed_threshold = changed
                        else:
                            moved = max(0, min(999, query['original_wrong_axis'] + changed - query['original_threshold']))
                            history[len(prompt) + query['wrong_axis_index']] = COORD0 + moved
                            observed_threshold = query['original_threshold']
                        logits = _native_logits(qwen, batch.inputs, history, forwards)
                        observed = int(logits.argmax().item())
                        if axis == 'same' and changed == query['original_threshold']:
                            baseline_logits = logits
                            if first_baseline is None:
                                identity = _native_logits(qwen, batch.inputs, history, forwards)
                                identity_max = float((logits - identity).abs().max().item())
                                selected_lp = float(torch.log_softmax(logits, 0)[query['observed_token_id']].item())
                                native_lp = query.get('saved_selected_logprob')
                                greedy_equal = observed == query['observed_token_id']
                                write_json(ROOT / 'qualification' / 'native-replay-v1.json',
                                           {'case': qname, 'identity_full_vocabulary_max': identity_max,
                                            'saved_selected_logprob': native_lp, 'replay_selected_logprob': selected_lp,
                                            'selected_logprob_abs_diff': None if native_lp is None else abs(selected_lp - native_lp),
                                            'greedy_winner_equal_saved': greedy_equal,
                                            'tolerance': 2e-4, 'forwards_including_identity': forwards[0]})
                                if identity_max > 2e-4 or not greedy_equal or (native_lp is not None and abs(selected_lp - native_lp) > 2e-4):
                                    raise ValueError('first native source replay or identity qualification failed')
                                first_baseline = qname
                        read = readout(logits, coord, observed_threshold)
                        conditions.append({'axis': axis, 'requested_threshold': changed,
                                           'actual_changed_bin': changed if axis == 'same' else moved,
                                           'observed_threshold': observed_threshold,
                                           'reasons': spec['reasons'], 'history_sha256': digest_json(history),
                                           'readout': read})
                if baseline_logits is None:
                    raise ValueError('baseline absent from frozen intervention set')
                base = next(x for x in conditions if x['axis'] == 'same' and x['requested_threshold'] == query['original_threshold'])
                frozen_probs = torch.tensor(base['readout']['coordinate_logprobs_conditional']).exp()
                for condition in conditions:
                    t = condition['observed_threshold']
                    null = float(frozen_probs[:t + 1].sum())
                    condition['frozen_logit_rethreshold_illegal_mass'] = null
                    condition['redistributed_illegal_mass'] = condition['readout']['illegal_family_mass'] - null
                write_json(qpath, {'schema': 'coordinate_order_knowledge.native_cell.v1',
                                   'case': case['image_id'], 'query': {k: v for k, v in query.items() if k not in ('history_token_ids', 'prefix_token_ids')},
                                   'prompt_sha256': case['prompt_sha256'], 'conditions': conditions})
        # Keep text-only interface separate from the native-task panel. Batch to
        # avoid hundreds of needless short source forwards.
        tokenizer = qwen.tokenizer
        labels = [tokenizer.encode(x, add_special_tokens=False) for x in ('A', 'B')]
        if any(len(x) != 1 for x in labels) or labels[0] == labels[1]:
            raise ValueError('text-assay answer labels are not distinct one-token labels')
        all_text = []
        for pair_index, (a, b) in enumerate(plan['textual_pairs']):
            for order in ('ascending', 'descending'):
                x, y = (a, b) if order == 'ascending' else (b, a)
                for placement in ('A_yes', 'B_yes'):
                    yes = placement[0]
                    for representation in ('coord_token', 'ordinary_decimal'):
                        ids = _text_prompt(tokenizer, x, y, representation=representation, yes_label=yes)
                        all_text.append({'pair_index': pair_index, 'pair': [a, b], 'input_order': order,
                                         'answer_placement': placement, 'representation': representation,
                                         'correct_label': yes if x > y else ('B' if yes == 'A' else 'A'),
                                         'prompt_ids': ids, 'prompt_sha256': digest_json(ids)})
        for offset in range(0, len(all_text), 8):
            _guard(start, forwards[0], ROOT)
            batch_rows = all_text[offset:offset + 8]
            old = tokenizer.padding_side
            tokenizer.padding_side = 'left'
            try:
                encoded = tokenizer.pad({'input_ids': [x['prompt_ids'] for x in batch_rows]}, padding=True, return_tensors='pt')
            finally:
                tokenizer.padding_side = old
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.no_grad():
                logits = qwen.model(**encoded, logits_to_keep=1, use_cache=False).logits[:, -1, :].float()
            forwards[0] += 1
            for item, z in zip(batch_rows, logits, strict=True):
                l = torch.log_softmax(z, -1)
                result = {k: v for k, v in item.items() if k != 'prompt_ids'}
                result.update({'A_logprob_full': float(l[labels[0][0]]), 'B_logprob_full': float(l[labels[1][0]]),
                               'winner_id': int(z.argmax()), 'label_token_ids': [labels[0][0], labels[1][0]]})
                write_json(cells / f'text-{item["pair_index"]:02d}-{item["input_order"]}-{item["answer_placement"]}-{item["representation"]}.json', result)
        status = 'complete'
    except Exception as exc:
        status = 'HOLD'
        error = repr(exc)
        raise
    finally:
        wall = time.time() - start
        write_json(ROOT / 'execution' / 'terminal-v1.json',
                   {'status': status, 'error': error, 'pid': os.getpid(), 'wall_start': start,
                    'wall_terminal': time.time(), 'model_wall_seconds': wall,
                    'allocated_gpu_seconds': wall, 'gpu_count': 1, 'model_forwards': forwards[0],
                    'cell_files': len(list(cells.glob('*.json')))})
    return ROOT / 'execution' / 'terminal-v1.json'


def reduce(plan_path: Path, output: Path) -> Path:
    plan = json.loads(plan_path.read_text())
    cells = ROOT / 'cells'
    per_query = []
    missing_native = []
    teacher_hold = []
    native_expected = 0
    for case in plan['cases']:
        for item in case['teacher']:
            if item.get('status') == 'HOLD':
                teacher_hold.append({'image_id': case['image_id'], **item})
        for query in [*case['queries'], *(x for x in case['teacher'] if x.get('kind') == 'teacher')]:
            native_expected += 1
            name = f'{case["image_id"]}-{query["kind"]}-{query["role"]}-{query["query_index"]}.json'
            path = cells / name
            if not path.is_file():
                missing_native.append(name)
                continue
            cell = json.loads(path.read_text())
            if cell['case'] != case['image_id'] or cell['query']['prefix_sha256'] != query['prefix_sha256']:
                raise ValueError(f'saved native cell has wrong frozen identity: {path}')
            cond = cell['conditions']
            base = next(x for x in cond if x['axis'] == 'same' and x['requested_threshold'] == query['original_threshold'])
            base_lp = base['readout']['coordinate_logprobs_conditional']
            base_mass = base['readout']['coordinate_family_mass']
            curve = []
            by_axis = {(x['axis'], x['requested_threshold']): x for x in cond}
            for x in cond:
                lp = x['readout']['coordinate_logprobs_conditional']
                v = query['observed_successor']
                candidates = [k for k in (v + 1, v + 2) if k <= 999 and k > x['observed_threshold']]
                neighbor = sum(lp[k] - base_lp[k] for k in candidates) / len(candidates) if candidates else None
                tv = sum(abs(math.exp(p) - math.exp(q)) for p, q in zip(lp, base_lp, strict=True)) / 2
                curve.append({'axis': x['axis'], 'requested_threshold': x['requested_threshold'],
                              'actual_changed_bin': x['actual_changed_bin'],
                              'observed_threshold': x['observed_threshold'],
                              'full_winner_id': x['readout']['full_vocabulary_winner'],
                              'legal_full_winner': (x['readout']['full_vocabulary_winner'] - COORD0 > x['observed_threshold']
                                                    if COORD0 <= x['readout']['full_vocabulary_winner'] < COORD0 + 1000 else None),
                              'family_mass': x['readout']['coordinate_family_mass'],
                              'family_mass_delta': x['readout']['coordinate_family_mass'] - base_mass,
                              'illegal_family_mass': x['readout']['illegal_family_mass'],
                              'frozen_logit_rethreshold_illegal_mass': x['frozen_logit_rethreshold_illegal_mass'],
                              'redistributed_illegal_mass': x['redistributed_illegal_mass'],
                              'candidate_logprob_delta': lp[v] - base_lp[v],
                              'still_legal_neighbor_logprob_delta': neighbor,
                              'candidate_minus_neighbor_delta': (lp[v] - base_lp[v] - neighbor) if neighbor is not None else None,
                              'total_variation_from_original': tv,
                              'equality_bin_probability': x['readout']['equality_bin_probability']})
            crossing = None
            v = query['observed_successor']
            if 1 <= v <= 998 and ('same', v - 1) in by_axis and ('same', v) in by_axis:
                legal = by_axis['same', v - 1]
                illegal = by_axis['same', v]
                wrong = by_axis.get(('wrong', v))
                ll = legal['readout']['coordinate_logprobs_conditional']
                il = illegal['readout']['coordinate_logprobs_conditional']
                wl = wrong['readout']['coordinate_logprobs_conditional'] if wrong else None
                neighbors = [k for k in (v + 1, v + 2) if k <= 999]
                crossing = {'candidate_bin': v, 'candidate_logprob_illegal_minus_legal': il[v] - ll[v],
                            'still_legal_neighbor_mean_illegal_minus_legal': sum(il[k] - ll[k] for k in neighbors) / len(neighbors),
                            'candidate_selectivity': (il[v] - ll[v]) - sum(il[k] - ll[k] for k in neighbors) / len(neighbors),
                            'wrong_axis_candidate_delta_from_original': None if wl is None else wl[v] - base_lp[v],
                            'wrong_axis_actual_changed_bin': None if wrong is None else wrong['actual_changed_bin']}
            per_query.append({'image_id': case['image_id'], 'case_status': case['status'],
                              'history_kind': query['kind'], 'role': query['role'],
                              'original_role_illegal': query['original_role_illegal'],
                              'original_threshold': query['original_threshold'], 'observed_successor': v,
                              'query_index': query['query_index'], 'source_native_kind': query.get('source_native_kind'),
                              'annotation_owner_id': query.get('annotation_owner_id'), 'curve': curve, 'crossing': crossing,
                              'cell_binding': binding(path)})
    textual = []
    missing_text = []
    for pair_index in range(64):
        for order in ('ascending', 'descending'):
            for placement in ('A_yes', 'B_yes'):
                for representation in ('coord_token', 'ordinary_decimal'):
                    name = f'text-{pair_index:02d}-{order}-{placement}-{representation}.json'
                    path = cells / name
                    if not path.is_file():
                        missing_text.append(name)
                        continue
                    data = json.loads(path.read_text())
                    if data['pair'] != plan['textual_pairs'][pair_index] or data['input_order'] != order or data['answer_placement'] != placement or data['representation'] != representation:
                        raise ValueError(f'text cell differs from frozen pair/order/interface: {path}')
                    correct = data[f'{data["correct_label"]}_logprob_full']
                    wrong_label = 'B' if data['correct_label'] == 'A' else 'A'
                    data['correct_minus_wrong_logprob'] = correct - data[f'{wrong_label}_logprob_full']
                    textual.append(data)
    text_summary = {}
    for representation in ('coord_token', 'ordinary_decimal'):
        rows = [x for x in textual if x['representation'] == representation]
        text_summary[representation] = {'planned': 256, 'observed': len(rows),
                                        'correct_sign': sum(x['correct_minus_wrong_logprob'] > 0 for x in rows),
                                        'mean_correct_minus_wrong_logprob': sum(x['correct_minus_wrong_logprob'] for x in rows) / len(rows) if rows else None}
    result = {'schema': 'coordinate_order_knowledge.reduction.v1', 'plan': binding(plan_path),
              'terminal': binding(ROOT / 'execution' / 'terminal-v1.json'),
              'native_queries': {'planned': native_expected, 'observed': len(per_query), 'HOLD': missing_native,
                                 'teacher_correspondence_HOLD': teacher_hold},
              'textual': {'planned': 512, 'observed': len(textual), 'HOLD': missing_text,
                          'summary': text_summary, 'rows': textual},
              'per_query': per_query,
              'observed_native_by_status_role': {f'{kind}:{role}': count for (kind, role), count in
                   Counter((x['history_kind'], x['role']) for x in per_query).items()}}
    write_json(output, result)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'run', 'reduce'))
    parser.add_argument('--plan', type=Path, default=ROOT / 'selection' / 'plan-v1.json')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', type=Path, default=ROOT / 'reduction-v1.json')
    args = parser.parse_args()
    print(prepare() if args.command == 'prepare' else run(args.plan, args.device) if args.command == 'run' else reduce(args.plan, args.output))


if __name__ == '__main__':
    main()
