"""Ten B16 cached owner-cue paths and an isolated fixed-hidden FP32 companion."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
import traceback
from pathlib import Path

from probes import box_continuity as continuity, coordinate_readout as readout
from probes.rule_stability import artifacts as a, data

ROOT, PREVIOUS = a.ROOT, readout.PREVIOUS
UNIT = ROOT / 'research/experiments/2026-10-04-owner-entry-localization'
OUTPUT = ROOT / 'outputs/research/physical-fn-recovery/2026-10-04/owner-entry-localization'
SCHEMA = 'owner-entry-localization-v1'
COORD_START, COORD_IDS = readout.COORD_START, readout.COORD_IDS
SOURCE_PATHS = ['probes/coordinate_diagnostics/__init__.py', 'probes/coordinate_diagnostics/owner_entry.py',
                'tests/probes/coordinate_diagnostics/test_owner_entry.py']
BOUNDS = dict(requests=10, actions=841, checkpoint_loads=1, gpu=0, process_count=1,
    maximum_actions_per_request=236, maximum_context_tokens=1598, optimizer=0, backward=0,
    replay=0, training=0, warmup=0, exports=0, shadow_states=40, wall_seconds=900,
    rss_bytes=32 * 1024**3, cuda_allocated_bytes=12 * 1024**3, retained_bytes=64 * 1024**2)
NATIVE_BOTTLE = [151646, 8987, 151647, 151648, 151670, 151683, 152206, 152660,
                 151649, 151646, 65, 62118, 151647, 151648]
CORRECTED_BOTTLE = list(NATIVE_BOTTLE)
CORRECTED_BOTTLE[5:8] = [151694, 152187, 152669]
OWNERS = {'context': dict(image_id=351017, ann_id=-1276804344338180, desc='person', box=[0, 24, 517, 999]),
          'target': dict(image_id=351017, ann_id=-4947389372712316, desc='bottle', box=[186, 30, 207, 106]),
          'control': dict(image_id=351017, ann_id=1489041, desc='bottle', box=[495, 396, 543, 614]),
          'person': dict(image_id=13348, ann_id=191150, desc='person', box=[546, 633, 556, 682])}
binding, revision = readout.binding, readout.revision


def producer_identity():
    diff = subprocess.check_output(['git', 'diff', 'HEAD', '--', *SOURCE_PATHS], cwd=ROOT)
    return dict(commit=revision(), diff_sha256=hashlib.sha256(diff).hexdigest(),
                files={p: a.digest(ROOT / p) for p in SOURCE_PATHS})


def definitions():
    rows = [('bottle-native', 'native', 'uncued', 'target'),
            ('person-native', 'native', 'uncued', 'person'),
            ('bottle-native-target-x1', 'native', 'x1_cued', 'target'),
            ('bottle-corrected-history', 'corrected', 'uncued', 'target'),
            ('bottle-corrected-target-x1', 'corrected', 'x1_cued', 'target'),
            ('bottle-corrected-control-x1', 'corrected', 'x1_cued', 'control'),
            ('bottle-target-teacher', 'corrected', 'teacher_path', 'target'),
            ('bottle-control-teacher', 'corrected', 'teacher_path', 'control'),
            ('person-target-x1', 'native', 'x1_cued', 'person'),
            ('person-target-teacher', 'native', 'teacher_path', 'person')]
    return [dict(condition=name, image_id=OWNERS[owner]['image_id'], history=history, population=pop,
                 owner=owner, mode='natural' if name.endswith('-native') else 'intervention',
                 budget=236 if owner == 'person' else 19, row_start=227 if owner == 'person' else 9,
                 observations=list(range(231, 235)) if owner == 'person' else list(range(14, 18)))
            for name, history, pop, owner in rows]


def forced_actions(cell):
    start = cell['observations'][0]
    prefix = (cell['expected_ids'][:start] if cell['image_id'] == 13348 else
              CORRECTED_BOTTLE if cell['history'] == 'corrected' else NATIVE_BOTTLE)
    forced = {str(i): token for i, token in enumerate(prefix)}
    count = 1 if cell['population'] == 'x1_cued' else 4 if cell['population'] == 'teacher_path' else 0
    forced.update({str(start + i): COORD_START + cell['target_box'][i] for i in range(count)})
    return forced


def validate_conditions(packet):
    if (packet.get('schema') != SCHEMA or packet.get('bounds') != BOUNDS or
            packet.get('coordinate_ids') != COORD_IDS or packet.get('model_loaded') is not False or
            packet.get('owners') != OWNERS or len(packet['conditions']) != 10):
        raise ValueError('frozen matrix/owners/resource contract differs')
    for cell, frozen in zip(packet['conditions'], definitions(), strict=True):
        if (any(cell.get(k) != v for k, v in frozen.items()) or
                cell['target_box'] != OWNERS[cell['owner']]['box'] or
                len(cell['expected_ids']) != cell['budget'] or cell['forced_actions'] != forced_actions(cell) or
                len(packet['requests'][str(cell['image_id'])]['prompt_token_ids']) != 1362 or
                set(cell['selectors']) != {str(i) for i in cell['observations']}):
            raise ValueError('literal selector/target/force/context differs')
        if cell['image_id'] == 351017 and cell['expected_ids'][:14] != NATIVE_BOTTLE:
            raise ValueError('saved bottle header differs')


def assemble_packet(q):
    """Tokenizer/frontend metadata only; reuse accepted immutable payload receipts."""
    from probes import iterative_positive as p
    from probes.rule_stability.__main__ import runtime_identity
    oldpath = continuity.OUTPUT / 'prepared-04/input-packet.json'
    old = a.load(oldpath)
    if q.model is not None or old['runtime'] != runtime_identity():
        raise ValueError('CPU preparation model/runtime differs')
    pipeline = dict(old['cached_pipeline_files'])
    pipeline['probes/box_continuity.py'] = a.digest(ROOT / 'probes/box_continuity.py')
    pipeline['src/qwen/untied_embeddings.py'] = a.digest(ROOT / 'src/qwen/untied_embeddings.py')
    for path, sha in pipeline.items():
        if a.digest(ROOT / path) != sha:
            raise ValueError('accepted cached source differs:' + path)
    full_images, manifest = data.load_inputs()
    requests = {str(r['image_id']): r for r in data.request_records(full_images, manifest) if r['image_id'] in (351017, 13348)}
    images = {str(i['image_id']): i for i in full_images if i['image_id'] in (351017, 13348)}
    if any(images[k] != old['images'][k] or requests[k] != old['requests'][k] for k in images):
        raise ValueError('original image/prompt/media differs')
    for key in images:
        readout.verify_frontend(q, images[key], requests[key])
    for role, owner in OWNERS.items():
        image = images[str(owner['image_id'])]
        matches = [o for o in image['objects'] if o['coco_ann_id'] == owner['ann_id']]
        if len(matches) != 1 or matches[0]['bbox_2d'] != owner['box'] or matches[0]['desc'] != owner['desc']:
            raise ValueError('selected annotation identity differs:' + role)
        _, sequence, _ = data.full_label_sequence(image, q)
        literal = '<|object_ref_start|>' + owner['desc'] + '<|object_ref_end|><|box_start|>'
        literal += ''.join(f'<|coord_{v}|>' for v in owner['box']) + '<|box_end|>'
        ids = q.tokenizer(literal, add_special_tokens=False)['input_ids']
        text = q.tokenizer.decode(sequence.input_ids, skip_special_tokens=False)
        if literal not in text or q.tokenizer.decode(ids, skip_special_tokens=False) != literal:
            raise ValueError('maintained label rendering/token identity differs')
    if len(images['351017']['objects']) != 49 or len(images['13348']['objects']) != 15:
        raise ValueError('annotation inventory differs')
    expected_header = '<|object_ref_start|>person<|object_ref_end|><|box_start|>'
    expected_header += ''.join(f'<|coord_{v}|>' for v in OWNERS['context']['box'])
    expected_header += '<|box_end|><|object_ref_start|>bottle<|object_ref_end|><|box_start|>'
    if q.tokenizer(expected_header, add_special_tokens=False)['input_ids'] != CORRECTED_BOTTLE:
        raise ValueError('corrected literal header token identity differs')
    keys = ['labels', 'manifest', 'policy', 'checkpoint', 'prior_acceptance', 'prior_readback', 'prior_qualification']
    bindings = {k: old['bindings'][k] for k in keys}
    bindings.update(protocol=binding(UNIT / 'unit.md'), prior_packet=binding(oldpath))
    for key in images:
        for name in ('raw', 'analysis', 'image'):
            bindings[f'{name}-{key}'] = old['bindings'][f'{name}-{key}']
    for b in bindings.values():
        if a.digest(b['path']) != b['sha256']:
            raise ValueError('qualified input changed:' + b['path'])
    continuity.check_payloads(old['payloads'])
    cells = definitions()
    for cell in cells:
        saved = a.load(bindings[f'raw-{cell["image_id"]}']['path'])
        analysis = a.load(bindings[f'analysis-{cell["image_id"]}']['path'])
        if saved['prompt_token_ids'] != requests[str(cell['image_id'])]['prompt_token_ids'] or q.tokenizer.decode(saved['token_ids'], skip_special_tokens=False) != saved['text']:
            raise ValueError('saved prompt/token/text differs')
        readout.verify_selected_rows(saved, analysis, 16, cell['image_id'])
        cell.update(expected_ids=saved['token_ids'][:cell['budget']], target_box=OWNERS[cell['owner']]['box'],
            saved_raw_logprobs=saved['raw_logprobs'][:cell['budget']],
            saved_policy_logprobs=saved['policy_logprobs'][:cell['budget']],
            selectors=readout.selectors(analysis, cell['observations']))
        cell['forced_actions'] = forced_actions(cell)
    packet = dict(schema=SCHEMA, bounds=BOUNDS, conditions=cells, coordinate_ids=COORD_IDS,
        owners=OWNERS, eos_id=old['eos_id'], object_start_id=old['object_start_id'], model_loaded=False,
        images=images, requests=requests, bindings=bindings, payloads=old['payloads'],
        runtime=runtime_identity(), producer=producer_identity(), cached_pipeline_files=pipeline)
    validate_conditions(packet)
    return packet


def prepare(directory):
    from probes import rollout_row_credit as retained
    packet = assemble_packet(retained.frontend())
    directory = Path(directory).resolve()
    if not directory.is_relative_to(OUTPUT):
        raise ValueError('preparation output outside unit owner')
    directory.mkdir(parents=True, exist_ok=False)
    a.write(directory / 'input-packet.json', packet)
    config = dict(schema=SCHEMA, released=False, source_revision=revision(), producer_files=packet['producer']['files'],
        input_packet=binding(directory / 'input-packet.json'), runtime=packet['runtime'], bounds=BOUNDS,
        output=str(OUTPUT / 'native-01'), retry='no_automatic_relaunch')
    a.write(directory / 'native-proposal.json', config)
    return config


def load_packet(config_path, *, cpu=False):
    from probes.rule_stability.__main__ import runtime_identity
    config = a.load(config_path)
    if config.get('schema') != SCHEMA or config.get('bounds') != BOUNDS or config.get('retry') != 'no_automatic_relaunch':
        raise ValueError('frozen release/resource contract differs')
    if a.digest(config['input_packet']['path']) != config['input_packet']['sha256']:
        raise ValueError('input packet changed')
    packet = a.load(config['input_packet']['path'])
    validate_conditions(packet)
    if config['runtime'] != runtime_identity() or config['runtime'] != packet['runtime']:
        raise ValueError('effective runtime differs')
    if config['producer_files'] != producer_identity()['files'] or config['producer_files'] != packet['producer']['files']:
        raise ValueError('producer bytes differ')
    if not cpu:
        if config.get('released') is not True or config.get('source_revision') != revision():
            raise ValueError('exact clean lead release required')
        if subprocess.check_output(['git', 'status', '--porcelain=v1', '--untracked-files=all'], cwd=ROOT):
            raise ValueError('native producer checkout is dirty')
        subprocess.run(['git', 'ls-files', '--error-unmatch', *SOURCE_PATHS], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
            raise ValueError('one serial GPU requires CUDA_VISIBLE_DEVICES=0')
    for b in packet['bindings'].values():
        if a.digest(b['path']) != b['sha256']:
            raise ValueError('bound input changed:' + b['path'])
    for path, sha in packet['cached_pipeline_files'].items():
        if a.digest(ROOT / path) != sha:
            raise ValueError('cached source changed:' + path)
    continuity.check_payloads(packet['payloads'])
    return config, packet


def target_metrics(scores, target):
    """Full native support ranks and within-coordinate ranks have distinct denominators."""
    import torch
    values = scores[0].float()
    coords = values[COORD_START:COORD_START + 1000]
    value = values[target]
    other = coords.clone()
    other[target - COORD_START] = -torch.inf
    full_other = values.clone()
    full_other[target] = -torch.inf
    return dict(target_token_id=target, coordinate_rank=1 + int((coords > value).sum()),
        full_rank=1 + int((values > value).sum()), coordinate_ties=int((coords == value).sum()),
        full_ties=int((values == value).sum()), target_minus_best_other_coordinate=float(value - other.max()),
        target_minus_best_other_full=float(value - full_other.max()))


class Observer(continuity.ContinuationProcessor):
    def __init__(self, *args, capture=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.capture = capture

    def __call__(self, input_ids, scores):
        self.last_history = input_ids[0, self.width:].tolist()
        if self.capture is not None:
            self.capture.bind(input_ids, scores)
        result = super().__call__(input_ids, scores)
        pos = len(self.steps) - 1
        if pos in self.cell['observations']:
            target = COORD_START + self.cell['target_box'][pos - self.cell['observations'][0]]
            self.observations[-1]['target_metrics'] = target_metrics(scores, target)
        return result


def runtime_rows(model, norm):
    """Detached snapshots of installed values, never a checkpoint or merged head."""
    import torch
    from src.qwen.untied_embeddings import SelectedDeltaOutputHead
    head = model.get_output_embeddings()
    if head is not norm.head or not isinstance(head, SelectedDeltaOutputHead) or type(head.base) is not torch.nn.Linear:
        raise ValueError('unsupported companion output-head composition')
    weight, delta = norm._validate_head()
    ids = norm.coordinate_ids.to(weight.device)
    rows = norm.coordinate_rows.to(delta.device)
    return dict(base=weight.detach().index_select(0, ids).cpu().clone(),
                delta=delta.detach().index_select(0, rows).cpu().clone())


class HeadCapture:
    """Observe original head input/output and pair it to the following raw processor."""
    def __init__(self, cell, width, transform, request_id):
        import torch
        self.cell, self.width, self.request_id = cell, width, request_id
        self.factors = transform.factors.detach().cpu().clone()
        if self.factors.dtype != torch.float64 or self.factors.shape != (1000,) or transform.coordinate_ids.tolist() != COORD_IDS:
            raise ValueError('companion median snapshot binding differs')
        self.calls, self.bound, self.pending = 0, 0, None
        self.states, self.metadata = [], []

    def hook(self, module, args, output):
        import torch
        if self.pending is not None or not args or args[0].ndim != 3 or args[0].shape[0] != 1 or output.ndim != 3 or output.shape[:2] != args[0].shape[:2]:
            raise ValueError('hidden/action head call alignment differs')
        h = args[0]
        if h.dtype not in (torch.bfloat16, torch.float32) or output.dtype != torch.bfloat16 or (self.calls > 0 and h.shape[1] != 1):
            raise ValueError('unsupported hidden/head dtype or cached singleton shape')
        pos = self.calls
        self.pending = dict(position=pos, head_input_shape=list(h.shape), hidden_dtype=str(h.dtype),
                            head_output_dtype=str(output.dtype))
        if pos in self.cell['observations']:
            self.pending.update(hidden=h[0, -1].detach().cpu().clone(),
                native_raw=output[0, -1, COORD_START:COORD_START + 1000].detach().cpu().clone())
        self.calls += 1
        return None

    def bind(self, input_ids, scores):
        import torch
        pos = input_ids.shape[1] - self.width
        if self.pending is None or self.pending['position'] != pos or self.bound != pos or input_ids.shape[0] != 1:
            raise ValueError('shifted hidden/action/prefix binding')
        if pos in self.cell['observations']:
            state = self.pending
            if not torch.equal(state['native_raw'].float(), scores[0, COORD_START:COORD_START + 1000].detach().cpu()):
                raise ValueError('native head result differs from incoming raw processor scores')
            self.states.append(dict(hidden=state.pop('hidden'), native_raw=state.pop('native_raw')))
            self.metadata.append(dict(state, request_id=self.request_id, prefix_token_ids=input_ids[0, self.width:].tolist()))
        self.pending = None
        self.bound += 1

    def finish(self, transform, tokens, median_observations):
        import torch
        if self.calls != len(tokens) or self.bound != len(tokens) or self.pending is not None:
            raise ValueError('unconsumed hidden/action capture')
        if not torch.equal(self.factors, transform.factors.detach().cpu()) or transform.coordinate_ids.tolist() != COORD_IDS:
            raise ValueError('median factors changed during acquisition')
        positions = [i for i in self.cell['observations'] if i < len(tokens)]
        if [m['position'] for m in self.metadata] != positions:
            raise ValueError('missing selected hidden capture')
        for state, meta, observation in zip(self.states, self.metadata, median_observations, strict=True):
            expected = (state['native_raw'].float().double() * self.factors).float()
            if meta['prefix_token_ids'] != tokens[:meta['position']] or expected.tolist() != observation['coordinate_scores']:
                raise ValueError('native median/capture/factor binding differs')
        return dict(status='captured', hook_calls=self.calls, metadata=self.metadata,
            factors=self.factors.tolist(), hidden=[s['hidden'].tolist() for s in self.states],
            native_head_coordinates=[s['native_raw'].tolist() for s in self.states],
            primary_untouched=True, same_original_hidden=True, hook_returns_none=True)


class NativeSession(readout.NativeSession):
    def __init__(self, checkpoint, packet):
        super().__init__(checkpoint, packet)
        try:
            self.companion_rows = runtime_rows(self.q.model, self.norm)
            self.companion_status = dict(status='supported', primary_untouched=True,
                operation='separate runtime BF16 base and FP32 output delta; no output-head DoRA')
        except ValueError as error:
            self.companion_rows = None
            self.companion_status = dict(status='HOLD', reason=str(error), primary_untouched=True)

    def generate(self, cell):
        import torch
        from src.qwen.generation import generate_continuations, NativeGenerationPolicy
        batch = self.batches[str(cell['image_id'])]
        width = len(batch.prompt_token_ids[0])
        raw = Observer(cell, self.packet, width)
        median = Observer(cell, self.packet, width, raw=raw)
        transform = self.norm.generation_transform()
        capture = HeadCapture(cell, width, transform, batch.request_ids[0]) if self.companion_rows is not None else None
        handle = self.q.model.get_output_embeddings().register_forward_hook(capture.hook) if capture else None
        raw.capture = capture
        self.failure_evidence, result = None, None
        begin = time.monotonic()
        try:
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                result = generate_continuations(self.q.model, batch, extensions=[()], budgets=[cell['budget']],
                    eos_token_id=self.packet['eos_id'], pad_token_id=self.q.tokenizer.pad_token_id,
                    policy=NativeGenerationPolicy(temperature=0, top_p=1, top_k=0, repetition_penalty=1, use_model_defaults=False),
                    trace='raw_and_policy', allow_pad_tokens=True, logits_processor=[raw, transform, median])[0]
            captured = capture.finish(transform, list(result.token_ids), median.observations) if capture else self.companion_status
        except Exception:
            self.failure_evidence = dict(condition=cell['condition'], request_id=batch.request_ids[0],
                phase='cached_generation_or_head_binding', budget=cell['budget'],
                confirmed_token_ids=list(result.token_ids) if result is not None else None,
                observed_cached_prefix=getattr(raw, 'last_history', []),
                processor_selections_emission_unconfirmed=raw.emitted,
                raw_steps=raw.steps, median_steps=median.steps,
                raw_observations=raw.observations, median_observations=median.observations)
            raise
        finally:
            if handle is not None:
                handle.remove()
        image = self.packet['images'][str(cell['image_id'])]
        return dict(request_id=batch.request_ids[0], width=image['width'], height=image['height'],
            token_ids=list(result.token_ids), text=self.q.tokenizer.decode(result.token_ids, skip_special_tokens=False),
            stop_reason=result.stop_reason, raw_logprobs=list(result.raw_logprobs), policy_logprobs=list(result.policy_logprobs),
            raw_steps=raw.steps, median_steps=median.steps, raw_observations=raw.observations,
            median_observations=median.observations, generation_seconds=time.monotonic() - begin,
            companion_capture=captured)


def validate_record(record, cell, packet):
    continuity.validate_record(record, cell, packet)
    for channel in ('raw', 'median'):
        for o in record[f'{channel}_observations']:
            target = COORD_START + cell['target_box'][o['position'] - cell['observations'][0]]
            m = o['target_metrics']
            if m['target_token_id'] != target or not (1 <= m['coordinate_rank'] <= 1000 and m['full_rank'] >= m['coordinate_rank'] and m['full_ties'] >= m['coordinate_ties'] >= 1):
                raise ValueError('target margin/rank alignment differs')
    cap = record['companion_capture']
    if cap['status'] == 'HOLD':
        if cap.get('primary_untouched') is not True:
            raise ValueError('companion changed primary')
    elif cap['status'] == 'captured':
        positions = [i for i in cell['observations'] if i < len(record['token_ids'])]
        if ([m['position'] for m in cap['metadata']] != positions or cap['hook_calls'] != len(record['token_ids']) or
                len(cap['hidden']) != len(positions) or len(cap['native_head_coordinates']) != len(positions) or
                len(cap['factors']) != 1000 or not cap['hook_returns_none'] or not cap['primary_untouched']):
            raise ValueError('companion capture support differs')
        for k, m in enumerate(cap['metadata']):
            if (m['request_id'] != record['request_id'] or m['prefix_token_ids'] != record['token_ids'][:m['position']] or
                    m['hidden_dtype'] not in ('torch.bfloat16', 'torch.float32') or m['head_output_dtype'] != 'torch.bfloat16' or
                    len(cap['hidden'][k]) != m['head_input_shape'][-1] or
                    cap['native_head_coordinates'][k] != record['raw_observations'][k]['coordinate_scores']):
                raise ValueError('companion hidden/head/action association differs')
    else:
        raise ValueError('unknown companion capture status')


def analyze(record, cell, packet, tokenizer):
    from probes.rule_stability.objectives import trajectory_analysis
    from src.eval.saved_rows import iou_xyxy
    image, designated = packet['images'][str(cell['image_id'])], OWNERS[cell['owner']]
    if any(record.get(k) != image[k] for k in ('width', 'height')):
        raise ValueError('parser image dimensions differ from bound original image')
    result = trajectory_analysis(record, tokenizer)
    rows, alignment = [], {}
    for row in result['rows']:
        start = row['positions'][0]
        homologous = start == cell['row_start'] and row['description'] == designated['desc']
        if homologous and row['coordinate_positions'] == cell['observations']:
            for slot, pos in enumerate(row['coordinate_positions']):
                alignment[str(pos)] = ['x1', 'y1', 'x2', 'y2'][slot]
        box = row['bbox']
        supplied = [p for p in row['coordinate_positions'] if str(p) in cell['forced_actions']]
        view = dict(order=row['order'], start_action=start, completion_action=row['completion_position'],
            description=row['description'], box=box, complete=True, valid=row['valid'],
            width=box[2] - box[0], height=box[3] - box[1], coordinate_positions=row['coordinate_positions'],
            coordinate_action_family=row['coordinate_action_family'], homologous=homologous,
            role=cell['population'] if homologous else 'supplied_context' if row['completion_position'] < cell['row_start'] else 'other',
            supplied_coordinate_actions=supplied, physical_recovery_credit=False,
            designated_annotation_id=designated['ann_id'], designated_category_matches=row['description'] == designated['desc'],
            designated_iou=iou_xyxy(box, designated['box']),
            same_category_overlaps=[dict(annotation_id=o['coco_ann_id'], bbox=o['bbox_2d'], iou=iou_xyxy(box, o['bbox_2d']))
                for o in image['objects'] if o['desc'] == row['description']],
            coordinate_errors=[s for s in result['geometry_sites'] if s.get('position') in row['positions']])
        rows.append(view)
    selected = next((r for r in rows if r['homologous']), None)
    return dict(rows=rows, selected_row=selected, selected_row_unavailable=None if selected else 'incomplete_or_semantically_diverged',
        alignment=alignment, malformed=result['malformed'], parser_status=result['parser_status'],
        burdens=result['burdens'], action_family_burdens=result['action_family_burdens'],
        geometry_sites=result['geometry_sites'], empty_legal_sets=result['empty_legal_sets'],
        actual_free_actions=sum(str(i) not in cell['forced_actions'] for i in range(len(record['token_ids']))),
        stop_reason=record['stop_reason'], emitted_eos=record['token_ids'][-1] == packet['eos_id'],
        censored_at_literal_budget=record['stop_reason'] == 'length')


def coordinate_summary(values, target_bin):
    import torch
    x = torch.tensor(values, dtype=torch.float32)
    if x.shape != (1000,) or not bool(torch.isfinite(x).all()):
        raise ValueError('companion coordinate support/nonfinite differs')
    best = sorted(range(1000), key=lambda i: (-values[i], i))[:5]
    other = x.clone()
    other[target_bin] = -torch.inf
    return dict(winner=best[0], top5=[dict(bin=i, score=values[i]) for i in best],
        winner_ties=int((x == x.max()).sum()), top2_margin=float(x[best[0]] - x[best[1]]),
        target_rank=1 + int((x > x[target_bin]).sum()), target_ties=int((x == x[target_bin]).sum()),
        target_minus_best_other=float(x[target_bin] - other.max()))


def shadow_scores(hidden, base, delta, factors):
    """Separate detached CPU IEEE-FP32 products; no native module invocation."""
    import torch
    from src.qwen.coordinate_policy import scale_coordinate_logits
    if (hidden.device.type != 'cpu' or base.device.type != 'cpu' or delta.device.type != 'cpu' or
            hidden.dtype not in (torch.bfloat16, torch.float32) or base.dtype != torch.bfloat16 or delta.dtype != torch.float32 or
            factors.dtype != torch.float64 or factors.device.type != 'cpu' or factors.shape != (1000,) or
            base.shape != delta.shape or base.shape != (1000, hidden.numel()) or hidden.ndim != 1 or
            any(t.requires_grad or not bool(torch.isfinite(t).all()) for t in (hidden, base, delta, factors))):
        raise ValueError('unsupported detached CPU companion composition')
    with torch.inference_mode(), torch.autocast('cpu', enabled=False), torch.backends.flags(fp32_precision='ieee'), torch.backends.mkldnn.flags(enabled=False, allow_tf32=None):
        base32 = hidden.float() @ base.float().t()
        delta32 = hidden.float() @ delta.t()
        raw = base32 + delta32
        median = scale_coordinate_logits(raw, torch.arange(1000), factors)
        precision = dict(device='cpu', generic_fp32=torch.backends.fp32_precision,
            mkldnn_enabled=torch.backends.mkldnn.enabled, autocast_enabled=torch.is_autocast_enabled('cpu'),
            hidden_dtype=str(hidden.dtype), base_runtime_dtype=str(base.dtype), delta_runtime_dtype=str(delta.dtype))
    return dict(base32=base32.tolist(), delta32=delta32.tolist(), raw=raw.tolist(), median=median.tolist(), precision=precision)


def companion_projection(output, packet, statuses, rows_binding, composition):
    """Run only after primary acquisition/session close; unsupported companion HOLD is separate."""
    if rows_binding is None:
        return dict(status='HOLD', reason=composition['reason'], primary_untouched=composition['primary_untouched'], projections=[])
    import torch
    from safetensors.torch import load_file
    rows = load_file(rows_binding['path'], device='cpu')
    projections = []
    for entry, cell in zip(statuses, packet['conditions'], strict=True):
        if entry['status'] != 'completed':
            continue
        record = a.load(Path(output) / entry['filename'])
        captured = record['companion_capture']
        if captured['status'] != 'captured':
            return dict(status='HOLD', reason='missing_capture', primary_untouched=True, projections=projections)
        factors = torch.tensor(captured['factors'], dtype=torch.float64)
        for index, meta in enumerate(captured['metadata']):
            pos = meta['position']
            if meta['request_id'] != record['request_id'] or meta['prefix_token_ids'] != record['token_ids'][:pos] or pos not in cell['observations']:
                raise ValueError('saved hidden/action/prefix association differs')
            dtype = torch.bfloat16 if meta['hidden_dtype'] == 'torch.bfloat16' else torch.float32
            h = torch.tensor(captured['hidden'][index], dtype=dtype)
            scores = shadow_scores(h, rows['base'], rows['delta'], factors)
            channels = {}
            for channel in ('raw', 'median'):
                native = next(o['coordinate_scores'] for o in record[f'{channel}_observations'] if o['position'] == pos)
                n = torch.tensor(native, dtype=torch.float64).softmax(-1)
                c = torch.tensor(scores[channel], dtype=torch.float64).softmax(-1)
                target = cell['target_box'][pos - cell['observations'][0]]
                channels[channel] = dict(native=coordinate_summary(native, target), companion=coordinate_summary(scores[channel], target),
                    conditional_TV=float((n - c).abs().sum() / 2),
                    conditional_W1_bins=float((n.cumsum(0)[:-1] - c.cumsum(0)[:-1]).abs().sum()))
            projections.append(dict(condition=cell['condition'], position=pos, capture_index=index,
                factors=captured['factors'], scores=scores, comparisons=channels))
    if len(projections) > 40:
        raise ValueError('companion state budget exceeded')
    return dict(status='complete', runtime_rows=rows_binding, composition=composition, projections=projections,
        arithmetic='CPU separate IEEE FP32 base/delta products and FP32 sum; captured FP64 factors multiply then FP32 cast',
        full_vocabulary_scores=False, decoder_calls=0, native_head_calls=0, feedback_into_generation=False)


def validate_companion(output, records, packet):
    """Saved companion cannot masquerade as primary/full-vocabulary or change its snapshot."""
    import torch
    result = a.load(Path(output) / 'companion.json')
    if result['status'] == 'HOLD':
        if not result['primary_untouched']:
            raise ValueError('companion changed primary computation')
        return result
    if result.get('feedback_into_generation') is not False or result.get('full_vocabulary_scores') is not False or result.get('decoder_calls') != 0 or result.get('native_head_calls') != 0:
        raise ValueError('companion fed generation or claimed full vocabulary')
    if a.digest(result['runtime_rows']['path']) != result['runtime_rows']['sha256']:
        raise ValueError('companion runtime rows changed')
    expected = {(name, o['position']) for name, r in records.items() for o in r['raw_observations']}
    if {(p['condition'], p['position']) for p in result['projections']} != expected or len(result['projections']) != len(expected):
        raise ValueError('companion capture/projection support differs')
    for p in result['projections']:
        record = records[p['condition']]
        cap = record['companion_capture']
        if p['factors'] != cap['factors'] or cap['metadata'][p['capture_index']]['position'] != p['position']:
            raise ValueError('companion shared hidden/factors changed')
        scores = p['scores']
        raw = torch.tensor(scores['base32'], dtype=torch.float32) + torch.tensor(scores['delta32'], dtype=torch.float32)
        scaled = (raw.double() * torch.tensor(p['factors'], dtype=torch.float64)).float()
        if raw.tolist() != scores['raw'] or scaled.tolist() != scores['median']:
            raise ValueError('companion separate projection/factor arithmetic differs')
    return result


def consume(output, packet, tokenizer):
    validate_conditions(packet)
    for b in packet['bindings'].values():
        if a.digest(b['path']) != b['sha256']:
            raise ValueError('saved consumer input changed')
    output = Path(output)
    index = a.load(output / 'conditions.json')
    if [e['condition'] for e in index] != [c['condition'] for c in packet['conditions']]:
        raise ValueError('consumer condition index differs')
    rows, passed, records = [], {}, {}
    observed = dict(x1_cued=0, uncued=0, teacher_path=0)
    for entry, cell in zip(index, packet['conditions'], strict=True):
        name, image = cell['condition'], cell['image_id']
        if entry['status'] == 'HOLD':
            rows.append(dict(condition=name, status='HOLD', reason=entry['reason'], population=cell['population']))
            continue
        if entry['status'] != 'completed' or (cell['mode'] != 'natural' and not passed.get(image)):
            raise ValueError('dependent artifact after natural fidelity HOLD')
        path = output / entry['filename']
        if path.name != name + '.json':
            raise ValueError('condition artifact filename differs')
        record = a.load(path)
        if record['schema'] != SCHEMA or record['condition'] != name or record['image_id'] != image:
            raise ValueError('record condition identity differs')
        validate_record(record, cell, packet)
        f = continuity.fidelity(record, cell) if cell['mode'] == 'natural' else None
        if f is not None:
            passed[image] = f['qualified']
        analysis = analyze(record, cell, packet, tokenizer)
        observed[cell['population']] += 1
        records[name] = record
        rows.append(dict(condition=name, status='completed', population=cell['population'], artifact=binding(path),
            natural_fidelity=f, generated_actions=len(record['token_ids']), analysis=analysis,
            observation_metrics={channel: [dict(position=o['position'], semantic_role=analysis['alignment'].get(str(o['position'])),
                **continuity.score_summary(o, record[f'{channel}_steps'][o['position']]), target_metrics=o['target_metrics'])
                for o in record[f'{channel}_observations']] for channel in ('raw', 'median')}))
    if {p.name for p in output.glob('bottle-*.json')} | {p.name for p in output.glob('person-*.json')} != {e['filename'] for e in index if e['status'] == 'completed'}:
        raise ValueError('unconsumed/extra condition artifact')
    return dict(schema=SCHEMA, complete=all(e['status'] == 'completed' for e in index) and all(passed.values()),
        conditions=rows, completed_requests=len(records), generated_actions=sum(len(r['token_ids']) for r in records.values()),
        denominators=dict(prescribed=dict(x1_cued=4, uncued=3, teacher_path=3), observed=observed),
        companion=validate_companion(output, records, packet),
        limitations='Supplied history/coordinates; conditional localization and literal teacher scores only; no physical recovery, prevalence, training cause or CE comparison.')


def resource_excess(output, begin, *, native):
    values = readout.usage(output, native=native)
    return values, [k for k, v in values.items() if v > BOUNDS[k]] + (['wall_seconds'] if time.monotonic() - begin > 900 else [])


def run(config_path, output, *, cpu_factory=None):
    output = Path(output).resolve()
    if not output.is_relative_to(OUTPUT):
        raise ValueError('invocation output outside unit owner')
    output.mkdir(parents=True, exist_ok=False)
    begin, session, packet, statuses, passed = time.monotonic(), None, None, [], {}
    rows_binding, companion_status = None, None
    native, status, code = cpu_factory is None, 'technical_HOLD', 2
    counts = dict(checkpoint_loads=0, fixture_sessions=0, attempted_requests=0, completed_requests=0,
        generated_actions=0, optimizer=0, backward=0, replay=0, training=0, warmup=0, exports=0)
    a.write(output / 'invocation.json', dict(schema=SCHEMA, config=binding(config_path), pid=os.getpid(),
        started=time.time(), compute='native' if native else 'CPU_FIXTURE', retry='no_automatic_relaunch'))
    try:
        config, packet = load_packet(config_path, cpu=not native)
        if native and str(output) != config['output']:
            raise ValueError('output differs from exact release')
        a.write(output / 'qualification.json', dict(source_revision=config['source_revision'], producer=config['producer_files'],
            runtime=config['runtime'], inputs=packet['bindings'], payloads=packet['payloads']))
        _, exceeded = resource_excess(output, begin, native=native)
        if exceeded:
            raise ValueError('resource excess before load:' + ','.join(exceeded))
        load_begin = time.monotonic()
        counts['checkpoint_loads' if native else 'fixture_sessions'] = 1
        session = (cpu_factory or NativeSession)(PREVIOUS / 'checkpoint-16', packet)
        a.write(output / 'load-B16.json', dict(seconds=time.monotonic() - load_begin, composition=session.composition))
        companion_status = session.companion_status
        if session.companion_rows is not None:
            from safetensors.torch import save_file
            path = output / 'companion-runtime-rows.safetensors'
            save_file(session.companion_rows, str(path))
            rows_binding = binding(path)
        for cell in packet['conditions']:
            if cell['mode'] != 'natural' and not passed.get(cell['image_id']):
                statuses.append(dict(condition=cell['condition'], status='HOLD', reason='natural_fidelity_dependency'))
                continue
            _, exceeded = resource_excess(output, begin, native=native)
            if exceeded:
                break
            counts['attempted_requests'] += 1
            record = session.generate(cell)
            record.update(schema=SCHEMA, condition=cell['condition'], image_id=cell['image_id'],
                likelihood_meaning=dict(raw='actual selected-action raw conditional', median_steps='unforced median selected-action',
                    policy='post-force: forced positions zero; free positions unforced median'))
            filename = cell['condition'] + '.json'
            a.write(output / filename, record)
            counts['completed_requests'] += 1
            counts['generated_actions'] += len(record['token_ids'])
            statuses.append(dict(condition=cell['condition'], status='completed', filename=filename))
            validate_record(record, cell, packet)
            if cell['mode'] == 'natural':
                f = continuity.fidelity(record, cell)
                passed[cell['image_id']] = f['qualified']
                a.write(output / f'fidelity-{cell["image_id"]}.json', f)
            a.write(output / f'resource-{counts["completed_requests"]:02d}.json',
                dict(seconds=time.monotonic() - begin, counts=counts, usage=resource_excess(output, begin, native=native)[0]))
        done = {e['condition'] for e in statuses}
        statuses.extend(dict(condition=c['condition'], status='HOLD', reason='resource_limit:' + ','.join(exceeded))
            for c in packet['conditions'] if c['condition'] not in done)
        a.write(output / 'conditions.json', statuses)
        tokenizer = session.q.tokenizer
        session.close()
        session = None
        companion_begin = time.monotonic()
        try:
            companion = companion_projection(output, packet, statuses, rows_binding, companion_status)
        except (ValueError, RuntimeError) as error:
            companion = dict(status='HOLD', reason=str(error), primary_untouched=True, projections=[])
        a.write(output / 'companion.json', companion)
        a.write(output / 'companion-cost.json', dict(seconds=time.monotonic() - companion_begin,
            states=len(companion['projections']), decoder_calls=0, native_head_calls=0))
        report = consume(output, packet, tokenizer)
        report['compute'] = 'native' if native else 'CPU_FIXTURE'
        a.write(output / 'readback.json', report)
        load_packet(config_path, cpu=not native)
        if report['complete'] and not resource_excess(output, begin, native=native)[1]:
            status, code = 'complete', 0
    except Exception as error:
        a.write(output / 'error.json', dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc()))
        if session is not None and getattr(session, 'failure_evidence', None) is not None:
            a.write(output / 'partial-generation.json', session.failure_evidence)
        if packet is not None and not (output / 'conditions.json').exists():
            done = {e['condition'] for e in statuses}
            statuses.extend(dict(condition=c['condition'], status='HOLD', reason=f'execution_failure:{type(error).__name__}')
                for c in packet['conditions'] if c['condition'] not in done)
            a.write(output / 'conditions.json', statuses)
    finally:
        if session is not None:
            try:
                session.close()
            except Exception as error:
                a.write(output / 'cleanup-error.json', dict(type=type(error).__name__, message=str(error)))
                status, code = 'technical_HOLD', 2
        values, exceeded = resource_excess(output, begin, native=native)
        a.write(output / 'terminal.json', dict(schema=SCHEMA, status=status, exit_code=code, counts=counts,
            seconds=time.monotonic() - begin, usage=values, resource_excess=exceeded,
            process_owner='single synchronous process; no background compute', cleanup='session references released',
            readback=binding(output / 'readback.json') if (output / 'readback.json').exists() else None))
    return code


def main(argv=None, *, cpu_factory=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'run', 'readback'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path)
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        result = prepare(args.output)
        print(json.dumps(dict(status='CPU_PREPARED', input_packet=result['input_packet'], model_loaded=False)))
        return 0
    if args.command == 'run':
        if args.config is None:
            parser.error('run needs --config')
        return run(args.config, args.output, cpu_factory=cpu_factory)
    invocation = a.load(args.output / 'invocation.json')
    if a.digest(invocation['config']['path']) != invocation['config']['sha256']:
        raise ValueError('saved invocation release changed')
    config = a.load(invocation['config']['path'])
    if a.digest(config['input_packet']['path']) != config['input_packet']['sha256']:
        raise ValueError('saved input packet changed')
    from probes import rollout_row_credit as retained
    report = consume(args.output, a.load(config['input_packet']['path']), retained.frontend().tokenizer)
    if report != {k: v for k, v in a.load(args.output / 'readback.json').items() if k != 'compute'}:
        raise ValueError('fresh saved consumer differs')
    print(json.dumps(dict(complete=report['complete'], completed_requests=report['completed_requests'],
        generated_actions=report['generated_actions'], denominators=report['denominators'])))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
