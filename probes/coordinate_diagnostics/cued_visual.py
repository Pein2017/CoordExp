"""Frozen x1-cued visual-use acquisition and saved-vector consumer."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from probes import box_continuity as continuity, coordinate_readout as readout
from probes.coordinate_diagnostics import owner_entry, visual_state as visual
from probes.rule_stability import artifacts as a

ROOT, PREVIOUS = a.ROOT, readout.PREVIOUS
UNIT = ROOT / 'research/experiments/2026-10-04-cued-visual-use'
OUTPUT = ROOT / 'outputs/research/physical-fn-recovery/2026-10-04/cued-visual-use'
SCHEMA = 'cued-visual-use-v1'
COORD_START, COORD_IDS = readout.COORD_START, readout.COORD_IDS
SOURCE_PATHS = ['probes/coordinate_diagnostics/cued_visual.py',
                'tests/probes/coordinate_diagnostics/test_cued_visual.py']
BOUNDS = dict(requests=12, actions=1530, checkpoint_loads=1, gpu=0, process_count=1,
    maximum_actions_per_request=236, maximum_context_tokens=1598, optimizer=0, backward=0,
    replay=0, training=0, warmup=0, exports=0, wall_seconds=900,
    rss_bytes=32 * 1024**3, cuda_allocated_bytes=12 * 1024**3, retained_bytes=128 * 1024**2)
SITES = {351017: dict(name='bottle', owner='target', position=14, budget=19, row_start=9),
         13348: dict(name='person', owner='person', position=231, budget=236, row_start=227)}
REFERENCES = {(351017, 'uncued'): 'bottle-native.json', (13348, 'uncued'): 'person-native.json',
              (351017, 'cued'): 'bottle-native-target-x1.json', (13348, 'cued'): 'person-target-x1.json'}
binding, revision = readout.binding, readout.revision


def producer_identity():
    diff = subprocess.check_output(['git', 'diff', 'HEAD', '--', *SOURCE_PATHS], cwd=ROOT)
    return dict(commit=revision(), diff_sha256=hashlib.sha256(diff).hexdigest(),
                files={p: a.digest(ROOT / p) for p in SOURCE_PATHS})


def definitions():
    order = [(351017, 'uncued', 'clean'), (13348, 'uncued', 'clean'),
             (351017, 'cued', 'clean'), (13348, 'cued', 'clean')]
    order += [(image, mode, region) for image in SITES for region in ('target', 'background')
              for mode in ('uncued', 'cued')]
    return [dict(condition=f'{SITES[i]["name"]}-{mode}-{region}', image_id=i, mode=mode,
        input_region=region, owner=SITES[i]['owner'], position=SITES[i]['position'],
        budget=SITES[i]['budget'], row_start=SITES[i]['row_start'],
        population='supplied_x1' if mode == 'cued' else 'supplied_native_prefix',
        observations=list(range(SITES[i]['position'], SITES[i]['position'] + 4)))
        for i, mode, region in order]


def validate_conditions(packet):
    if (packet.get('schema') != SCHEMA or packet.get('bounds') != BOUNDS or
            packet.get('coordinate_ids') != COORD_IDS or packet.get('model_loaded') is not False or
            packet.get('summary_convention') != 'detached-original-FP32-vector_CPU-single-thread-FP32' or
            len(packet['conditions']) != 12):
        raise ValueError('frozen packet/matrix/observation contract differs')
    prior = a.load(packet['bindings']['visual_packet']['path'])
    for key in ('images', 'requests', 'pixels', 'processor', 'original_media'):
        if packet[key] != prior[key]:
            raise ValueError('reused frozen RGB/request/processor identity differs:' + key)
    for cell, declared in zip(packet['conditions'], definitions(), strict=True):
        if any(cell.get(k) != v for k, v in declared.items()):
            raise ValueError('frozen cell/order/cue/action budget differs')
        ref = a.load(packet['bindings']['reference-' + str(cell['image_id']) + '-' + cell['mode']]['path'])
        native = a.load(readout.saved_path(16, cell['image_id']))
        pos = cell['position']; target = owner_entry.OWNERS[cell['owner']]['box']
        forced = {str(i): token for i, token in enumerate(native['token_ids'][:pos])}
        if cell['mode'] == 'cued':
            forced[str(pos)] = COORD_START + target[0]
        if (cell['forced_actions'] != forced or cell['expected_ids'] != ref['token_ids'] or
                len(cell['expected_ids']) != cell['budget'] or cell['target_box'] != target or
                cell['reference_median_winners'] != [s['argmax'] for s in ref['median_steps']] or
                cell['reference_median_logprobs'] != [s['pre_force_logprob'] for s in ref['median_steps']] or
                cell['reference_raw_logprobs'] != ref['raw_logprobs'] or
                len(packet['requests'][str(cell['image_id'])]['clean']['prompt_token_ids']) != 1362):
            raise ValueError('literal native/cued reference/force boundary differs')
    if sum(c['budget'] for c in packet['conditions']) != 1530 or sum(len(c['forced_actions']) for c in packet['conditions']) != 1476:
        raise ValueError('frozen logical work differs')


def prepare(directory):
    from probes import rollout_row_credit as retained, iterative_positive as p
    from probes.rule_stability.__main__ import runtime_identity
    oldpath = visual.OUTPUT / 'prepared-01/input-packet.json'; old = a.load(oldpath)
    if old['runtime'] != runtime_identity():
        raise ValueError('qualified current runtime differs')
    for b in old['bindings'].values():
        if a.digest(b['path']) != b['sha256']:
            raise ValueError('qualified immutable input changed:' + b['path'])
    for path, sha in old['cached_pipeline_files'].items():
        if a.digest(ROOT / path) != sha:
            raise ValueError('qualified helper source changed:' + path)
    continuity.check_payloads(old['payloads']); visual.verify_pixels(old)
    q = retained.frontend()
    if q.model is not None:
        raise ValueError('CPU preparation loaded a model')
    for image, variants in old['requests'].items():
        for region, request in variants.items():
            batch = p.native_request(request, p.load(p.POLICY), q.processor)
            if visual.batch_identity(batch) != old['processor'][image][region]:
                raise ValueError('reused actual processor/text/grid/media differs')
    bindings = dict(old['bindings'], protocol=binding(UNIT / 'unit.md'), visual_packet=binding(oldpath))
    cells = definitions()
    for cell in cells:
        path = owner_entry.OUTPUT / 'native-01' / REFERENCES[(cell['image_id'], cell['mode'])]
        bindings['reference-' + str(cell['image_id']) + '-' + cell['mode']] = binding(path)
        ref = a.load(path); pos = cell['position']
        native = a.load(readout.saved_path(16, cell['image_id']))
        cell.update(expected_ids=ref['token_ids'], target_box=owner_entry.OWNERS[cell['owner']]['box'],
            reference_median_winners=[s['argmax'] for s in ref['median_steps']],
            reference_median_logprobs=[s['pre_force_logprob'] for s in ref['median_steps']],
            reference_raw_logprobs=ref['raw_logprobs'],
            forced_actions={str(i): token for i, token in enumerate(native['token_ids'][:pos])})
        if cell['mode'] == 'cued':
            cell['forced_actions'][str(pos)] = COORD_START + cell['target_box'][0]
    pipeline = dict(old['cached_pipeline_files'])
    for path in ('probes/coordinate_diagnostics/visual_state.py', 'probes/box_continuity.py',
                 'probes/coordinate_readout.py', 'probes/coordinate_diagnostics/owner_entry.py'):
        pipeline[path] = a.digest(ROOT / path)
    packet = dict(schema=SCHEMA, conditions=cells, bounds=BOUNDS, bindings=bindings,
        cached_pipeline_files=pipeline, runtime=old['runtime'], payloads=old['payloads'],
        model_loaded=False, coordinate_ids=COORD_IDS, eos_id=old['eos_id'], object_start_id=old['object_start_id'],
        vocabulary_size=len(q.tokenizer), summary_convention='detached-original-FP32-vector_CPU-single-thread-FP32',
        producer=producer_identity(), **{k: old[k] for k in ('images', 'requests', 'pixels', 'processor', 'original_media')})
    validate_conditions(packet)
    directory = Path(directory).resolve()
    if not directory.is_relative_to(OUTPUT):
        raise ValueError('preparation outside task owner')
    directory.mkdir(parents=True, exist_ok=False)
    a.write(directory / 'input-packet.json', packet)
    proposal = dict(schema=SCHEMA, released=False, source_revision=revision(), producer_files=packet['producer']['files'],
        input_packet=binding(directory / 'input-packet.json'), runtime=packet['runtime'], bounds=BOUNDS,
        output=str(OUTPUT / 'native-01'), retry='no_automatic_relaunch')
    a.write(directory / 'native-proposal.json', proposal)
    return proposal


def load_packet(config_path, *, cpu=False):
    from probes.rule_stability.__main__ import runtime_identity
    config = a.load(config_path)
    if config.get('schema') != SCHEMA or config.get('bounds') != BOUNDS or config.get('retry') != 'no_automatic_relaunch':
        raise ValueError('release/resource contract differs')
    if a.digest(config['input_packet']['path']) != config['input_packet']['sha256']:
        raise ValueError('input packet changed')
    packet = a.load(config['input_packet']['path']); validate_conditions(packet)
    if config['runtime'] != runtime_identity() or config['runtime'] != packet['runtime']:
        raise ValueError('effective runtime differs')
    if config['producer_files'] != producer_identity()['files'] or config['producer_files'] != packet['producer']['files']:
        raise ValueError('producer bytes differ')
    if not cpu:
        if config.get('released') is not True or config.get('source_revision') != revision():
            raise ValueError('exact clean lead release required')
        if subprocess.check_output(['git', 'status', '--porcelain=v1', '--untracked-files=all'], cwd=ROOT):
            raise ValueError('native source is dirty')
        subprocess.run(['git', 'ls-files', '--error-unmatch', *SOURCE_PATHS], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
            raise ValueError('single GPU0 requires CUDA_VISIBLE_DEVICES=0')
    for b in packet['bindings'].values():
        if a.digest(b['path']) != b['sha256']:
            raise ValueError('bound input changed:' + b['path'])
    for path, sha in packet['cached_pipeline_files'].items():
        if a.digest(ROOT / path) != sha:
            raise ValueError('cached source changed:' + path)
    continuity.check_payloads(packet['payloads']); visual.verify_pixels(packet)
    return config, packet


@contextmanager
def cpu_diagnostics():
    """One task-local CPU reduction convention, never a native decision setting."""
    import torch
    before = torch.get_num_threads()
    try:
        if before != 1:
            torch.set_num_threads(1)
        with torch.autocast('cpu', enabled=False):
            yield
    finally:
        if before != 1:
            torch.set_num_threads(before)


@cpu_diagnostics()
def canonical(vector, packet, target):
    import torch
    if vector.device.type != 'cpu' or vector.dtype != torch.float32 or vector.shape != (packet['vocabulary_size'],):
        raise ValueError('canonical summary requires original detached CPU FP32 full vector')
    return dict(**readout.capture_scores(vector[None], packet),
                target_metrics=owner_entry.target_metrics(vector[None], target))


class Observer(continuity.ContinuationProcessor):
    """The maintained native decision path; only diagnostics use saved CPU values."""
    def __init__(self, *args, channel, **kwargs):
        super().__init__(*args, **kwargs)
        self.channel, self.vectors = channel, {}

    def __call__(self, input_ids, scores):
        position = input_ids.shape[1] - self.width
        vector = scores[0].detach().float().cpu().clone() if position in self.cell['observations'] else None
        result = super().__call__(input_ids, scores)
        if vector is not None:
            previous = self.observations[-1]; key = f'{self.channel}_{position}'
            target = COORD_START + self.cell['target_box'][position - self.cell['position']]
            self.vectors[key] = vector
            self.observations[-1] = dict(position=position, prefix_token_ids=previous['prefix_token_ids'],
                vector_key=key, native_device_full_log_normalizer=previous['full_log_normalizer'],
                native_device=str(scores.device), **canonical(vector, self.packet, target))
        return result


class NativeSession(visual.NativeSession):
    def generate(self, cell):
        import torch
        from src.qwen.generation import generate_continuations, NativeGenerationPolicy
        batch = self.batches[(str(cell['image_id']), cell['input_region'])]
        width = len(batch.prompt_token_ids[0])
        raw = Observer(cell, self.packet, width, channel='raw')
        median = Observer(cell, self.packet, width, raw=raw, channel='median')
        begin = time.monotonic()
        self.failure_evidence = None
        try:
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                result = generate_continuations(self.q.model, batch, extensions=[()], budgets=[cell['budget']],
                    eos_token_id=self.packet['eos_id'], pad_token_id=self.q.tokenizer.pad_token_id,
                    policy=NativeGenerationPolicy(temperature=0, top_p=1, top_k=0, repetition_penalty=1, use_model_defaults=False),
                    trace='raw_and_policy', allow_pad_tokens=True,
                    logits_processor=[raw, self.norm.generation_transform(), median])[0]
        except Exception:
            self.failure_evidence = dict(condition=cell['condition'], selected_unconfirmed=raw.emitted,
                raw_steps=raw.steps, median_steps=median.steps,
                raw_observations=raw.observations, median_observations=median.observations)
            raise
        image = self.packet['images'][str(cell['image_id'])]
        return dict(request_id=batch.request_ids[0], width=image['width'], height=image['height'],
            token_ids=list(result.token_ids), text=self.q.tokenizer.decode(result.token_ids, skip_special_tokens=False),
            stop_reason=result.stop_reason, raw_logprobs=list(result.raw_logprobs), policy_logprobs=list(result.policy_logprobs),
            raw_steps=raw.steps, median_steps=median.steps, raw_observations=raw.observations,
            median_observations=median.observations, generation_seconds=time.monotonic() - begin,
            _tensors=dict(raw.vectors, **median.vectors))


def validate_record(record, cell, packet):
    import torch
    from safetensors.torch import load_file
    continuity.validate_record(record, cell, packet)
    request = packet['requests'][str(cell['image_id'])][cell['input_region']]
    image = packet['images'][str(cell['image_id'])]
    if (record.get('schema') != SCHEMA or record.get('condition') != cell['condition'] or
            record.get('image_id') != cell['image_id'] or record.get('request_id') != request['request_id'] or
            any(record.get(k) != image[k] for k in ('width', 'height'))):
        raise ValueError('saved cell/request identity differs')
    b = record['score_tensors']
    if a.digest(b['path']) != b['sha256']:
        raise ValueError('saved score vector bytes changed')
    tensors = load_file(b['path'], device='cpu')
    if {k: visual.tensor_identity(v) for k, v in tensors.items()} != record['tensor_identities']:
        raise ValueError('saved original FP32 score identity differs')
    keys = set()
    for channel in ('raw', 'median'):
        for obs in record[channel + '_observations']:
            pos = obs['position']; key = f'{channel}_{pos}'; keys.add(key)
            if obs['vector_key'] != key:
                raise ValueError('score/action association differs')
            compact = canonical(tensors[key], packet, COORD_START + cell['target_box'][pos - cell['position']])
            if any(obs.get(k) != v for k, v in compact.items()):
                raise ValueError('canonical CPU summary differs from saved original vector')
    if keys != set(tensors):
        raise ValueError('extra/missing selected score vectors')


@cpu_diagnostics()
def fidelity(record, cell, packet):
    mismatch = [i for i in range(cell['budget']) if i >= len(record['token_ids']) or
        record['token_ids'][i] != cell['expected_ids'][i] or
        record['median_steps'][i]['argmax'] != cell['reference_median_winners'][i]]
    ref = a.load(packet['bindings']['reference-' + str(cell['image_id']) + '-' + cell['mode']]['path'])
    comparisons = {channel: [dict(position=o['position'], **continuity.score_difference(old, o))
        for o in record[channel + '_observations'] for old in ref[channel + '_observations']
        if o['position'] == old['position']] for channel in ('raw', 'median')}
    return dict(qualified=not mismatch, mismatch_positions=mismatch,
        raw_logprob_max_abs_difference=max((abs(x-y) for x,y in zip(record['raw_logprobs'], cell['reference_raw_logprobs'])), default=None),
        median_logprob_max_abs_difference=max((abs(x['pre_force_logprob']-y) for x,y in zip(record['median_steps'], cell['reference_median_logprobs'])), default=None),
        saved_coordinate_scores=comparisons, numeric_comparisons_descriptive=True,
        cued_x1_preforce_winner_is_reference_not_required_cue=True)


def dependencies(cell, qualified):
    if cell['input_region'] == 'clean':
        return []
    return [f'{SITES[cell["image_id"]]["name"]}-{mode}-clean' for mode in ('uncued', 'cued')
            if not qualified.get(f'{SITES[cell["image_id"]]["name"]}-{mode}-clean', False)]


def causal_roles(record, cell):
    pos = cell['position']; ids = record['token_ids']
    return {str(i): ('x1', 'y1', 'x2', 'y2')[i-pos]
        for i in cell['observations'] if i < len(ids) and ids[:pos] == cell['expected_ids'][:pos]
        and all(token in COORD_IDS for token in ids[pos:i])}


@cpu_diagnostics()
def compare(left, right):
    result = []
    for channel in ('raw', 'median'):
        left_obs = {o['position']:o for o in left['record'][channel + '_observations']}
        for obs in right['record'][channel + '_observations']:
            pos = obs['position']; other = left_obs.get(pos)
            role = right['causal_roles'].get(str(pos))
            if other is None or role is None or left['causal_roles'].get(str(pos)) != role:
                result.append(dict(channel=channel, position=pos, status='unavailable_semantic_alignment'))
                continue
            matched = other['prefix_token_ids'] == obs['prefix_token_ids']
            result.append(dict(channel=channel, position=pos, role=role, prefix_exact=matched,
                interpretation='matched_literal_prefix' if matched else 'different_generated_prefix',
                **continuity.score_difference(other, obs)))
    return result


@cpu_diagnostics()
def contrasts(rows):
    present = {r['condition']:r for r in rows if r['status']=='completed' and r['control']['qualified']}
    result = []
    for image, site in SITES.items():
        for mode in ('uncued', 'cued'):
            names = [f'{site["name"]}-{mode}-{region}' for region in ('clean', 'target', 'background')]
            if any(n not in present for n in names):
                result.append(dict(image_id=image, mode=mode, status='HOLD', unavailable=[n for n in names if n not in present]));continue
            clean, target, background = [present[n] for n in names]
            values = [r['analysis']['selected_row']['designated_iou'] if r['analysis']['selected_row'] else None for r in (clean,target,background)]
            behavior = dict(clean_iou=values[0], target_iou=values[1], background_iou=values[2],
                target_minus_clean=None if None in values[:2] else values[1]-values[0],
                background_minus_clean=None if values[0] is None or values[2] is None else values[2]-values[0],
                target_minus_background=None if values[1] is None or values[2] is None else values[1]-values[2])
            pairs = {label:compare(left,right) for label,left,right in
                [('target_clean',clean,target),('background_clean',clean,background),('target_background',background,target)]}
            y1 = []
            if mode == 'cued':
                pos = site['position']+1
                for channel in ('raw','median'):
                    observations = [next((o for o in r['record'][channel+'_observations'] if o['position']==pos),None) for r in (clean,target,background)]
                    if any(o is None for o in observations) or any(r['causal_roles'].get(str(pos))!='y1' for r in (clean,target,background)):
                        y1.append(dict(channel=channel,status='unavailable_first_free_y1'));continue
                    if not observations[0]['prefix_token_ids']==observations[1]['prefix_token_ids']==observations[2]['prefix_token_ids']:
                        raise ValueError('cued first-free y1 is not the same literal prefix')
                    q = [o['coordinate_scores'][owner_entry.OWNERS[site['owner']]['box'][1]]-o['coordinate_scores'][0] for o in observations]
                    y1.append(dict(channel=channel,position=pos,prefix_exact=True,clean_q=q[0],
                        target_minus_clean=q[1]-q[0],background_minus_clean=q[2]-q[0],target_minus_background=q[1]-q[2],
                        comparisons={k:next(x for x in v if x.get('channel')==channel and x.get('position')==pos) for k,v in pairs.items()}))
            divergence = {label:next((i for i,(x,y) in enumerate(zip(left['record']['token_ids'],right['record']['token_ids'])) if x!=y),
                min(len(left['record']['token_ids']),len(right['record']['token_ids'])) if len(left['record']['token_ids'])!=len(right['record']['token_ids']) else None)
                for label,left,right in [('target_clean',clean,target),('background_clean',clean,background),('target_background',background,target)]}
            result.append(dict(image_id=image,mode=mode,status='available',behavior=behavior,
                first_free_divergence=divergence,first_free_y1=y1,coordinate_comparisons=pairs))
    return result


@cpu_diagnostics()
def consume(output, packet, tokenizer):
    validate_conditions(packet);visual.verify_pixels(packet)
    for b in packet['bindings'].values():
        if a.digest(b['path']) != b['sha256']:
            raise ValueError('saved consumer input changed')
    output = Path(output);index = a.load(output/'conditions.json');rows=[];qualified={};files=set()
    if [r['condition'] for r in index] != [c['condition'] for c in packet['conditions']]:
        raise ValueError('saved consumer matrix/order differs')
    for entry,cell in zip(index,packet['conditions'],strict=True):
        name=cell['condition'];unmet=dependencies(cell,qualified)
        if entry['status']=='skipped-HOLD':
            if not unmet and not entry['reason'].startswith(('resource_limit','execution_failure')):
                raise ValueError('unjustified dependent HOLD')
            rows.append(entry);continue
        if entry['status']=='attempted-invalid':
            if not (output/entry['failure']).is_file():raise ValueError('missing invalid-cell evidence')
            files.update(entry[k] for k in ('filename','failure') if k in entry);qualified[name]=False;rows.append(entry);continue
        if entry['status']!='completed' or unmet or entry['filename']!=name+'.json':
            raise ValueError('published invalid dependency/cell')
        files.add(entry['filename']);record=a.load(output/entry['filename']);validate_record(record,cell,packet)
        if tokenizer.decode(record['token_ids'],skip_special_tokens=False)!=record['text']:
            raise ValueError('original full decode differs')
        control=fidelity(record,cell,packet) if cell['input_region']=='clean' else dict(qualified=True,quality_gate=False)
        qualified[name]=control['qualified'];analysis=owner_entry.analyze(record,cell,packet,tokenizer)
        selected=analysis['selected_row']
        if selected:
            selected['best_same_category_overlap']=max(selected['same_category_overlaps'],key=lambda x:x['iou'],default=None)
        summaries={channel:[dict(position=o['position'],role=causal_roles(record,cell).get(str(o['position'])),
            **continuity.score_summary(o,record[channel+'_steps'][o['position']]),target_metrics=o['target_metrics'],
            fixed_GT_minus_zero=o['coordinate_scores'][cell['target_box'][o['position']-cell['position']]]-o['coordinate_scores'][0])
            for o in record[channel+'_observations']] for channel in ('raw','median')}
        rows.append(dict(condition=name,status='completed',image_id=cell['image_id'],mode=cell['mode'],input_region=cell['input_region'],
            control=control,analysis=analysis,causal_roles=causal_roles(record,cell),scores=summaries,record=record,
            artifact=binding(output/entry['filename'])))
    actual={p.name for stem in ('bottle','person') for p in output.glob(stem+'-*.json')}
    if files!=actual:raise ValueError('extra/unconsumed cell or partial artifact')
    result=contrasts(rows)
    return dict(schema=SCHEMA,complete=len([r for r in rows if r['status']=='completed'])==12 and all(qualified.values()),
        conditions=rows,contrasts=result,control_qualified=qualified,requested_cells=12,
        requested_denominators=dict(states=2,modes=2,image_variants=3,clean_anchors=4,cued_cells=6,uncued_cells=6),
        attempted_requests=sum(e['status']!='skipped-HOLD' for e in index),completed_requests=sum(e['status']=='completed' for e in index),
        generated_actions=sum(len(r['record']['token_ids']) for r in rows if r['status']=='completed'),
        limitations='Supplied native history/x1; original annotation references after masking. No physical recovery, unique owner, population, layer, attention-path or training claim.')


def resources(output, begin, native):
    values=readout.usage(output,native=native)
    return values,[k for k,v in values.items() if v>BOUNDS[k]]+(['wall_seconds'] if time.monotonic()-begin>BOUNDS['wall_seconds'] else [])


def run(config_path,output,*,cpu_factory=None):
    import torch
    from safetensors.torch import save_file
    output=Path(output).resolve()
    if not output.is_relative_to(OUTPUT):raise ValueError('invocation outside task owner')
    output.mkdir(parents=True,exist_ok=False);begin=time.monotonic();session=None;packet=None;native=cpu_factory is None
    status,code='technical_HOLD',2;entries=[];qualified={}
    counts=dict(checkpoint_loads=0,fixture_sessions=0,attempted_requests=0,completed_requests=0,generated_actions=0,
        optimizer=0,backward=0,replay=0,training=0,warmup=0,exports=0)
    a.write(output/'invocation.json',dict(schema=SCHEMA,config=binding(config_path),pid=os.getpid(),started=time.time(),
        cpu_threads_incoming=torch.get_num_threads(),compute='native' if native else 'CPU_FIXTURE'))
    try:
        config,packet=load_packet(config_path,cpu=not native)
        if native and str(output)!=config['output']:raise ValueError('output differs from exact release')
        a.write(output/'qualification.json',dict(source_revision=config['source_revision'],producer=config['producer_files'],runtime=config['runtime'],inputs=packet['bindings'],cached_pipeline_files=packet['cached_pipeline_files'],payloads=packet['payloads']))
        _,exceeded=resources(output,begin,native)
        if exceeded:raise ValueError('resource excess before load:'+','.join(exceeded))
        start=time.monotonic();counts['checkpoint_loads' if native else 'fixture_sessions']=1
        session=(cpu_factory or NativeSession)(PREVIOUS/'checkpoint-16',packet)
        a.write(output/'load-B16.json',dict(seconds=time.monotonic()-start,composition=session.composition))
        for cell in packet['conditions']:
            name=cell['condition'];unmet=dependencies(cell,qualified)
            if unmet:entries.append(dict(condition=name,status='skipped-HOLD',reason='dependency:'+','.join(unmet)));continue
            _,exceeded=resources(output,begin,native)
            if exceeded:break
            counts['attempted_requests']+=1
            try:
                record=session.generate(cell);record.update(schema=SCHEMA,condition=name,image_id=cell['image_id'])
                tensors=record.pop('_tensors');path=output/(name+'.safetensors');save_file(tensors,str(path))
                record.update(score_tensors=binding(path),tensor_identities={k:visual.tensor_identity(v) for k,v in tensors.items()})
                a.write(output/(name+'.json'),record);counts['generated_actions']+=len(record['token_ids'])
                validate_record(record,cell,packet)
                check=fidelity(record,cell,packet) if cell['input_region']=='clean' else dict(qualified=True,quality_gate=False)
                qualified[name]=check['qualified'];counts['completed_requests']+=1
                entries.append(dict(condition=name,status='completed',filename=name+'.json'))
                a.write(output/('control-'+name+'.json'),check)
            except ValueError as error:
                qualified[name]=False;failure=name+'-invalid.json'
                a.write(output/failure,dict(error=str(error),phase='acquisition_or_saved_binding',attempted_requests=counts['attempted_requests'],
                    partial=getattr(session,'failure_evidence',None)))
                entry=dict(condition=name,status='attempted-invalid',failure=failure,reason=str(error))
                if (output/(name+'.json')).exists():entry['filename']=name+'.json'
                entries.append(entry)
            a.write(output/f'resource-{counts["attempted_requests"]:02d}.json',dict(seconds=time.monotonic()-begin,counts=counts,usage=resources(output,begin,native)[0]))
        done={e['condition'] for e in entries}
        entries.extend(dict(condition=c['condition'],status='skipped-HOLD',reason='resource_limit:'+','.join(exceeded)) for c in packet['conditions'] if c['condition'] not in done)
        a.write(output/'conditions.json',entries);tokenizer=session.q.tokenizer;session.close();session=None
        report=consume(output,packet,tokenizer);report['compute']='native' if native else 'CPU_FIXTURE'
        a.write(output/'readback.json',report);load_packet(config_path,cpu=not native)
        if report['complete'] and not resources(output,begin,native)[1]:status,code='complete',0
    except Exception as error:
        a.write(output/'error.json',dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
        if session is not None and getattr(session,'failure_evidence',None) is not None:
            a.write(output/'partial-generation.json',session.failure_evidence)
        if packet is not None and not (output/'conditions.json').exists():
            done={e['condition'] for e in entries}
            entries.extend(dict(condition=c['condition'],status='skipped-HOLD',reason='execution_failure:'+type(error).__name__) for c in packet['conditions'] if c['condition'] not in done)
            a.write(output/'conditions.json',entries)
    finally:
        if session is not None:
            try:session.close()
            except Exception as error:a.write(output/'cleanup-error.json',dict(message=str(error)));status,code='technical_HOLD',2
        usage,exceeded=resources(output,begin,native)
        a.write(output/'terminal.json',dict(status=status,exit_code=code,counts=counts,seconds=time.monotonic()-begin,
            usage=usage,resource_excess=exceeded,cleanup='single synchronous session references released',
            readback=binding(output/'readback.json') if (output/'readback.json').exists() else None))
    return code


def main(argv=None,*,cpu_factory=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--output',required=True)
    p=sub.add_parser('run');p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('readback');p.add_argument('--output',required=True)
    args=parser.parse_args(argv)
    if args.command=='prepare':print(json.dumps(prepare(args.output),sort_keys=True));return 0
    if args.command=='run':return run(args.config,args.output,cpu_factory=cpu_factory)
    from probes import rollout_row_credit as retained
    import torch
    print('cued_visual_cpu_threads_incoming='+str(torch.get_num_threads()),file=sys.stderr)
    output=Path(args.output);invocation=a.load(output/'invocation.json')
    config_path=invocation['config']['path']
    if a.digest(config_path)!=invocation['config']['sha256']:raise ValueError('saved config differs')
    _,packet=load_packet(config_path,cpu=invocation['compute']=='CPU_FIXTURE')
    q=retained.frontend()
    if q.model is not None:raise ValueError('saved consumer loaded a model')
    print(json.dumps(consume(output,packet,q.tokenizer),sort_keys=True));return 0


if __name__=='__main__':
    raise SystemExit(main())
