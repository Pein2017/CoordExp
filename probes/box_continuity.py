"""Thirteen B16 cached box continuations; no training or additional model queries."""
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

from probes import coordinate_readout as readout
from probes.rule_stability import artifacts as a, data

ROOT, PREVIOUS = a.ROOT, readout.PREVIOUS
UNIT = ROOT / 'research/experiments/2026-10-04-box-continuity'
OUTPUT = ROOT / 'outputs/research/physical-fn-recovery/2026-10-04/box-continuity'
SCHEMA = 'box-continuity-v1'
COORD_START, COORD_IDS = readout.COORD_START, readout.COORD_IDS
SOURCE_PATHS = ['probes/box_continuity.py', 'tests/probes/test_box_continuity.py']
BOUNDS = dict(requests=13, actions=3390, checkpoint_loads=1, gpu=0, process_count=1,
    maximum_actions_per_request=424, maximum_context_tokens=1744, optimizer=0, backward=0,
    replay=0, training=0, warmup=0, exports=0, wall_seconds=900, rss_bytes=32 * 1024**3,
    cuda_allocated_bytes=12 * 1024**3, retained_bytes=32 * 1024**2)
SITES = {13348: dict(name='normal', position=231, value=544, budget=236, row_order=25),
         7511: dict(name='zero-width', position=419, value=684, budget=424, row_order=46),
         351017: dict(name='bottle', position=16, value=23, budget=30, row_order=1)}
binding, revision = readout.binding, readout.revision


def producer_identity():
    diff = subprocess.check_output(['git', 'diff', 'HEAD', '--', *SOURCE_PATHS], cwd=ROOT)
    return dict(commit=revision(), diff_sha256=hashlib.sha256(diff).hexdigest(),
                files={path: a.digest(ROOT / path) for path in SOURCE_PATHS})


def definitions():
    cells = []
    def add(image, mode, sign=0):
        site = SITES[image]
        cells.append(dict(condition=f'B16/{site["name"]}/{mode}/{sign:+d}', image_id=image,
            mode=mode, sign=sign, edit_position=site['position'], original=COORD_START + site['value'],
            edited=COORD_START + site['value'] + sign, budget=site['budget'],
            clamp_positions=[site['position'] + 1, site['position'] + 2] if mode == 'clamped' else [],
            observations=list(range(14 if image == 351017 else site['position'], site['budget']))))
    for image in SITES:
        add(image, 'natural')
    for image in SITES:
        for sign in (-1, 1):
            add(image, 'free', sign)
            if image != 351017:
                add(image, 'clamped', sign)
    assert len(cells) == 13 and sum(c['budget'] for c in cells) == 3390
    return cells


def forced_actions(cell):
    forced = {str(i): token for i, token in enumerate(cell['expected_ids'][:cell['edit_position'] + 1])}
    forced[str(cell['edit_position'])] = cell['edited']
    forced.update({str(i): cell['expected_ids'][i] for i in cell['clamp_positions']})
    return forced


def validate_conditions(packet):
    if (packet.get('schema') != SCHEMA or packet.get('bounds') != BOUNDS or
            packet.get('coordinate_ids') != COORD_IDS or packet.get('model_loaded') is not False):
        raise ValueError('frozen packet/resource contract differs')
    if len(packet['conditions']) != 13:
        raise ValueError('frozen thirteen-cell matrix differs')
    for cell, declared in zip(packet['conditions'], definitions(), strict=True):
        if any(cell.get(k) != v for k, v in declared.items()):
            raise ValueError('frozen selector/edit/force/budget differs')
        if (len(cell['expected_ids']) != cell['budget'] or
                cell['expected_ids'][cell['edit_position']] != cell['original'] or
                cell['forced_actions'] != forced_actions(cell) or
                set(cell['selectors']) != {str(i) for i in cell['observations']} or
                len(packet['requests'][str(cell['image_id'])]['prompt_token_ids']) + cell['budget'] > 1744):
            raise ValueError('literal actions/context alignment differs')


def prepare(directory):
    """Reuse qualified immutable payloads; only tokenizer/processor computation."""
    from probes import rollout_row_credit as retained, iterative_positive as p
    from probes.rule_stability.__main__ import runtime_identity
    previous = a.load(readout.OUTPUT / 'prepared-01/input-packet.json')
    accepted = a.load(readout.OUTPUT / 'lead-consumer-check-01.json')
    if (accepted['status'] != 'lead-accepted' or
            a.digest(accepted['readback']['path']) != accepted['readback']['sha256'] or
            previous['runtime'] != runtime_identity()):
        raise ValueError('accepted readout/runtime identity differs')
    # These exact operators own cached score, normalization and loading semantics.
    pipeline = ['probes/coordinate_readout.py', 'src/qwen/coordinate_policy.py',
                'src/qwen/generation.py', 'probes/rule_stability/runner.py']
    for path in pipeline:
        old = subprocess.check_output(['git', 'show', f'{accepted["execution_commit"]}:{path}'], cwd=ROOT)
        if hashlib.sha256(old).hexdigest() != a.digest(ROOT / path):
            raise ValueError(f'accepted cached processor source differs: {path}')
    full_images, manifest = data.load_inputs()
    requests = {str(r['image_id']): r for r in data.request_records(full_images, manifest) if r['image_id'] in SITES}
    images = {str(i['image_id']): i for i in full_images if i['image_id'] in SITES}
    if images != previous['images'] or requests != previous['requests']:
        raise ValueError('original image/prompt/media bindings differ')
    q = retained.frontend()
    if q.model is not None:
        raise ValueError('CPU preparation loaded a model')
    for image, request in requests.items():
        readout.verify_frontend(q, images[image], request)
    bindings = dict(protocol=binding(UNIT / 'unit.md'), labels=binding(data.FULL_LABEL_PATH),
        manifest=binding(data.INPUT_MANIFEST_PATH), policy=binding(p.POLICY),
        prior_packet=binding(readout.OUTPUT / 'prepared-01/input-packet.json'),
        prior_acceptance=binding(readout.OUTPUT / 'lead-consumer-check-01.json'),
        prior_readback=binding(readout.OUTPUT / 'native-01/readback.json'),
        prior_qualification=binding(readout.OUTPUT / 'native-01/qualification.json'))
    saved, analyses = {}, {}
    for image in SITES:
        key = str(image)
        rawpath, analysispath = readout.saved_path(16, image), readout.saved_path(16, image, analysis=True)
        saved[key], analyses[key] = a.load(rawpath), a.load(analysispath)
        readout.verify_selected_rows(saved[key], analyses[key], 16, image)
        if (q.tokenizer.decode(saved[key]['token_ids'], skip_special_tokens=False) != saved[key]['text'] or
                saved[key]['prompt_token_ids'] != requests[key]['prompt_token_ids']):
            raise ValueError('saved original token/text/prompt identity differs')
        for name, path in [('raw', rawpath), ('analysis', analysispath), ('image', images[key]['image_path'])]:
            bindings[f'{name}-{image}'] = binding(path)
    bindings['checkpoint'] = binding(PREVIOUS / 'checkpoint-16/checkpoint.json')
    for name in ('B16-natural-351017', 'B16-copy--1-351017', 'B16-copy-+1-351017'):
        bindings[name] = binding(readout.OUTPUT / f'native-01/{name}.json')
        accepted_row = next(r for r in a.load(accepted['readback']['path'])['conditions']
                            if r.get('artifact', {}).get('path') == bindings[name]['path'])
        if bindings[name] != accepted_row['artifact']:
            raise ValueError('accepted reused cell bytes differ')
    for name, item in previous['bindings'].items():
        if name in ('labels', 'manifest', 'policy', 'checkpoint-16', 'saved-16/13348',
                    'saved-16/7511', 'saved-16/351017', 'analysis-16/13348',
                    'analysis-16/7511', 'analysis-16/351017', 'image-13348', 'image-7511', 'image-351017'):
            if a.digest(item['path']) != item['sha256']:
                raise ValueError('qualified original input changed')
    payloads = {path: item for path, item in previous['payloads'].items() if '/checkpoint-0/' not in path}
    check_payloads(payloads)
    meta = a.load(bindings['checkpoint']['path'])
    if any(meta.get(k) != v for k, v in dict(schema=a.SCHEMA, arm='B', version=16, engine='native').items()):
        raise ValueError('B16 checkpoint/schema identity differs')
    owner = previous['trusted_owner']
    matches = [o for o in images['13348']['objects'] if o['coco_ann_id'] == 191150]
    if (owner['ann_id'] != 191150 or owner['bbox'] != [546, 633, 556, 682] or
            len(matches) != 1 or matches[0]['bbox_2d'] != owner['bbox'] or matches[0]['desc'] != 'person'):
        raise ValueError('supplied normal owner differs')
    cells = definitions()
    for cell in cells:
        reference = saved[str(cell['image_id'])]
        cell.update(expected_ids=reference['token_ids'][:cell['budget']],
            saved_raw_logprobs=reference['raw_logprobs'][:cell['budget']],
            saved_policy_logprobs=reference['policy_logprobs'][:cell['budget']],
            selectors=readout.selectors(analyses[str(cell['image_id'])], cell['observations']))
        cell['forced_actions'] = forced_actions(cell)
    packet = dict(schema=SCHEMA, bounds=BOUNDS, conditions=cells, coordinate_ids=COORD_IDS,
        eos_id=previous['eos_id'], object_start_id=previous['object_start_id'], model_loaded=False,
        images=images, requests=requests, bindings=bindings, payloads=payloads, runtime=runtime_identity(),
        producer=producer_identity(), trusted_owner=owner,
        cached_pipeline_files={path: a.digest(ROOT / path) for path in pipeline},
        reuse_conditions=[c for c in previous['conditions'] if c['kind'] in ('natural', 'copy--1', 'copy-+1')
                          and c['image_id'] == 351017 and c['version'] == 16])
    validate_conditions(packet)
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


def check_payloads(payloads):
    for path, identity in payloads.items():
        stat = Path(path).stat()
        if stat.st_size != identity['size_bytes'] or stat.st_mtime_ns != identity['mtime_ns']:
            raise ValueError(f'qualified unchanged payload differs: {path}')


def load_packet(config_path, *, cpu=False):
    from probes.rule_stability.__main__ import runtime_identity
    config = a.load(config_path)
    if (config.get('schema') != SCHEMA or config.get('bounds') != BOUNDS or
            config.get('retry') != 'no_automatic_relaunch'):
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
    for identity in packet['bindings'].values():
        if a.digest(identity['path']) != identity['sha256']:
            raise ValueError(f'bound input changed: {identity["path"]}')
    for path, sha in packet['cached_pipeline_files'].items():
        if a.digest(ROOT / path) != sha:
            raise ValueError('cached processor source changed')
    check_payloads(packet['payloads'])
    for cell in packet['conditions']:
        saved = a.load(readout.saved_path(16, cell['image_id']))
        analysis = a.load(readout.saved_path(16, cell['image_id'], analysis=True))
        if (saved['token_ids'][:cell['budget']] != cell['expected_ids'] or
                readout.selectors(analysis, cell['observations']) != cell['selectors']):
            raise ValueError('saved reference/selector changed')
    return config, packet


class ContinuationProcessor:
    """Observe before forcing; raw selection follows the policy's actual action."""
    def __init__(self, cell, packet, prompt_width, *, raw=None):
        self.cell, self.packet, self.width, self.raw = cell, packet, prompt_width, raw
        self.steps, self.observations, self.emitted = [], [], []
        self.pending = None

    def select(self, token):
        import torch
        scores = self.pending
        if scores is None:
            raise ValueError('raw observation must precede policy selection')
        step = self.steps[-1]
        step.update(requested_token_id=token, pre_force_logprob=float(torch.log_softmax(scores, -1)[0, token]),
            selected_rank=1 + int((scores[0] > scores[0, token]).sum()),
            selected_tie_count=int((scores[0] == scores[0, token]).sum()))
        self.emitted.append(token)
        self.pending = None

    def __call__(self, input_ids, scores):
        import torch
        position = input_ids.shape[1] - self.width
        history = input_ids[0, self.width:].tolist()
        if (input_ids.shape[0] != 1 or scores.shape[0] != 1 or position != len(self.steps) or
                position >= self.cell['budget'] or history != self.emitted or
                any(history[int(i)] != token for i, token in self.cell['forced_actions'].items() if int(i) < position)):
            raise ValueError('cached singleton history/forcing alignment differs')
        if not bool(torch.isfinite(scores).all()):
            raise ValueError('nonfinite incoming score')
        forced = self.cell['forced_actions'].get(str(position))
        self.steps.append(dict(position=position, argmax=int(scores[0].argmax()), forced=forced is not None))
        self.pending = scores.float().clone()
        top = torch.topk(self.pending[0], 2)
        self.steps[-1].update(full_top2_margin=float(top.values[0] - top.values[1]),
                             full_winner_ties=int((self.pending[0] == top.values[0]).sum()))
        if position in self.cell['observations']:
            self.observations.append(dict(position=position, prefix_token_ids=history, **readout.capture_scores(scores, self.packet)))
        if self.raw is None:
            return scores
        selected = forced if forced is not None else int(scores[0].argmax())
        self.raw.select(selected)
        self.select(selected)
        if forced is None:
            return scores
        result = torch.full_like(scores, -torch.inf)
        result[0, selected] = 0
        return result


class NativeSession(readout.NativeSession):
    def generate(self, cell):
        import torch
        from src.qwen.generation import generate_continuations, NativeGenerationPolicy
        batch = self.batches[str(cell['image_id'])]
        raw = ContinuationProcessor(cell, self.packet, len(batch.prompt_token_ids[0]))
        median = ContinuationProcessor(cell, self.packet, len(batch.prompt_token_ids[0]), raw=raw)
        begin = time.monotonic()
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            result = generate_continuations(self.q.model, batch, extensions=[()], budgets=[cell['budget']],
                eos_token_id=self.packet['eos_id'], pad_token_id=self.q.tokenizer.pad_token_id,
                policy=NativeGenerationPolicy(temperature=0, top_p=1, top_k=0, repetition_penalty=1, use_model_defaults=False),
                trace='raw_and_policy', allow_pad_tokens=True,
                logits_processor=[raw, self.norm.generation_transform(), median])[0]
        image = self.packet['images'][str(cell['image_id'])]
        return dict(request_id=batch.request_ids[0], width=image['width'], height=image['height'],
            token_ids=list(result.token_ids), text=self.q.tokenizer.decode(result.token_ids, skip_special_tokens=False),
            stop_reason=result.stop_reason, raw_logprobs=list(result.raw_logprobs), policy_logprobs=list(result.policy_logprobs),
            raw_steps=raw.steps, median_steps=median.steps, raw_observations=raw.observations,
            median_observations=median.observations, generation_seconds=time.monotonic() - begin)


def validate_record(record, cell, packet):
    tokens = record['token_ids']
    if not tokens or len(tokens) > cell['budget']:
        raise ValueError('actual action budget differs')
    if any(tokens[int(i)] != token for i, token in cell['forced_actions'].items() if int(i) < len(tokens)):
        raise ValueError('actual forced action differs')
    eos = packet['eos_id']
    if eos in tokens[:-1] or record['stop_reason'] != ('im_end' if tokens[-1] == eos else 'length'):
        raise ValueError('actual EOS/stop differs')
    if len(tokens) != cell['budget'] and tokens[-1] != eos:
        raise ValueError('short nonterminal generation')
    for channel in ('raw', 'median'):
        steps, observations = record[f'{channel}_steps'], record[f'{channel}_observations']
        if len(steps) != len(tokens) or [s['position'] for s in steps] != list(range(len(tokens))):
            raise ValueError('every actual action requires aligned likelihood evidence')
        if [o['position'] for o in observations] != [i for i in cell['observations'] if i < len(tokens)]:
            raise ValueError('actual observation positions differ')
        for i, step in enumerate(steps):
            if (step['requested_token_id'] != tokens[i] or step['forced'] != (str(i) in cell['forced_actions']) or
                    not math.isfinite(step['pre_force_logprob']) or step['selected_rank'] < 1 or step['selected_tie_count'] < 1):
                raise ValueError('pre-force selected action/likelihood differs')
            if channel == 'median' and not step['forced'] and step['argmax'] != tokens[i]:
                raise ValueError('free action differs from median winner')
        for o in observations:
            values = [*o['coordinate_scores'], o['full_log_normalizer'], o['eos_score'], o['object_start_score'],
                      *(v['score'] for v in o['top5_noncoordinate'])]
            if (o['prefix_token_ids'] != tokens[:o['position']] or len(o['coordinate_scores']) != 1000 or
                    len(o['top5_noncoordinate']) != 5 or not all(math.isfinite(v) for v in values) or
                    any(v['token_id'] in COORD_IDS for v in o['top5_noncoordinate']) or
                    o['argmax'] != steps[o['position']]['argmax']):
                raise ValueError('compact score/prefix support differs')
    for key in ('raw_logprobs', 'policy_logprobs'):
        if len(record[key]) != len(tokens) or not all(math.isfinite(v) for v in record[key]):
            raise ValueError('actual selected-action traces differ')
    # Both owners use FP32 log_softmax; retain numeric deltas, reject action drift.
    for i, token in enumerate(tokens):
        if record['raw_steps'][i]['requested_token_id'] != token:
            raise ValueError('raw selected action differs')
        expected = 0. if str(i) in cell['forced_actions'] else record['median_steps'][i]['pre_force_logprob']
        if not math.isclose(record['policy_logprobs'][i], expected, rel_tol=0, abs_tol=2e-5):
            raise ValueError('forced/free policy trace bookkeeping differs')
        if not math.isclose(record['raw_logprobs'][i], record['raw_steps'][i]['pre_force_logprob'], rel_tol=0, abs_tol=2e-5):
            raise ValueError('raw trace bookkeeping differs')


def fidelity(record, cell):
    tokens = record['token_ids']
    mismatches = [dict(position=i, expected=token, actual=tokens[i] if i < len(tokens) else None,
                       winner=record['median_steps'][i]['argmax'] if i < len(tokens) else None)
        for i, token in enumerate(cell['expected_ids'])
        if i >= len(tokens) or tokens[i] != token or record['median_steps'][i]['argmax'] != token]
    return dict(qualified=not mismatches, mismatch_count=len(mismatches), mismatches=mismatches,
        raw_logprob_max_abs_difference=max(abs(x - y) for x, y in zip(record['raw_logprobs'], cell['saved_raw_logprobs'])),
        median_logprob_max_abs_difference=max(abs(s['pre_force_logprob'] - y)
            for s, y in zip(record['median_steps'], cell['saved_policy_logprobs'])),
        likelihood_differences_are_descriptive=True)


def score_summary(o, step):
    import torch
    scores = torch.tensor(o['coordinate_scores'], dtype=torch.float64)
    best = sorted(range(1000), key=lambda i: (-o['coordinate_scores'][i], i))[:5]
    value = step['requested_token_id'] - COORD_START
    return dict(coordinate_family_mass=math.exp(float(torch.logsumexp(scores, -1)) - o['full_log_normalizer']),
        coordinate_winner=best[0], top2_margin=float(scores[best[0]] - scores[best[1]]),
        coordinate_winner_ties=int((scores == scores.max()).sum()),
        top5_coordinates=[dict(bin=i, score=o['coordinate_scores'][i]) for i in best],
        full_winner=o['argmax'], top5_noncoordinate=o['top5_noncoordinate'],
        selected_token_id=step['requested_token_id'], selected_rank=step['selected_rank'],
        selected_tie_count=step['selected_tie_count'], unforced_selected_logprob=step['pre_force_logprob'], forced=step['forced'],
        full_top2_margin=step.get('full_top2_margin'), full_winner_ties=step.get('full_winner_ties'),
        selected_minus_adjacent_scores={str(i): float(scores[value] - scores[i])
            for i in (value - 1, value + 1) if 0 <= value < 1000 and 0 <= i < 1000})


def score_difference(left, right):
    import torch
    x = torch.tensor(left['coordinate_scores'], dtype=torch.float64)
    y = torch.tensor(right['coordinate_scores'], dtype=torch.float64)
    p, q = x.softmax(-1), y.softmax(-1)
    return dict(coordinate_conditional_TV=float((p - q).abs().sum() / 2),
        coordinate_conditional_W1_bins=float((p.cumsum(0)[:-1] - q.cumsum(0)[:-1]).abs().sum()),
        score_max_abs_difference=float((x - y).abs().max()), full_winner_changed=left['argmax'] != right['argmax'],
        exact_score_equal=left['coordinate_scores'] == right['coordinate_scores'],
        family_mass_left=math.exp(float(x.logsumexp(-1)) - left['full_log_normalizer']),
        family_mass_right=math.exp(float(y.logsumexp(-1)) - right['full_log_normalizer']))


def analyze(record, cell, packet, tokenizer):
    from probes.rule_stability.objectives import trajectory_analysis
    from src.eval.saved_rows import iou_xyxy
    image = packet['images'][str(cell['image_id'])]
    if any(record.get(key, image[key]) != image[key] for key in ('width', 'height')):
        raise ValueError('parser image dimensions differ from bound original image')
    parsed_record = dict(record, request_id=record.get('request_id', cell['condition']),
                         width=image['width'], height=image['height'])
    result = trajectory_analysis(parsed_record, tokenizer)
    selected, alignment = [], {}
    native = a.load(readout.saved_path(16, cell['image_id'], analysis=True))
    starts = {r['positions'][0]: r for r in native['rows']}
    for row in result['rows']:
        start = row['positions'][0]
        reference = starts.get(start)
        homologous = reference is not None and reference['description'] == row['description']
        if homologous and row['coordinate_positions']:
            for slot, position in enumerate(row['coordinate_positions']):
                alignment[str(position)] = [start, ('x1', 'y1', 'x2', 'y2')[slot], row['description']]
        if homologous:
            alignment[str(row['completion_position'])] = [start, 'closure', row['description']]
        if row['completion_position'] < cell['edit_position']:
            continue
        box = row['bbox']
        view = dict(row_order=row['order'], start_action=start, completion_action=row['completion_position'],
            description=row['description'], box=box, width=box[2] - box[0], height=box[3] - box[1],
            valid=row['valid'], complete=True, homologous=homologous,
            reference_order=reference['order'] if homologous else None, coordinate_positions=row['coordinate_positions'])
        if homologous:
            original = reference['bbox']
            view.update(displacement_bins=[x - y for x, y in zip(box, original)],
                center_displacement_bins=[(box[0] + box[2] - original[0] - original[2]) / 2,
                                          (box[1] + box[3] - original[1] - original[3]) / 2],
                width_change_bins=view['width'] - (original[2] - original[0]),
                height_change_bins=view['height'] - (original[3] - original[1]),
                freely_generated_corner_displacements={role: box[i] - original[i]
                    for i, (role, pos) in enumerate(zip(('x1', 'y1', 'x2', 'y2'), row['coordinate_positions']))
                    if str(pos) not in cell['forced_actions']})
        view['scale'] = (dict(denominator_bins=10, meaning='named GT width') if cell['image_id'] == 13348 else
                         dict(denominator_bins=23, meaning='generated native width proxy, not physical width')
                         if cell['image_id'] == 351017 else dict(denominator_bins=None, meaning='zero width: unavailable'))
        denominator = view['scale']['denominator_bins']
        view['corner_displacements_over_width'] = ([v / denominator for v in view['displacement_bins']]
            if homologous and denominator is not None else None)
        view['owner_annotation_id'] = 191150 if cell['image_id'] == 13348 and homologous and reference['order'] == 25 else None
        if view['owner_annotation_id'] is not None:
            owner = packet['trusted_owner']
            view['named_owner_iou'] = iou_xyxy(box, owner['bbox']) if row['valid'] else 0.
            alternatives = [dict(annotation_id=o['coco_ann_id'], category=o['desc'], iou=iou_xyxy(box, o['bbox_2d']))
                for o in packet['images']['13348']['objects'] if o['coco_ann_id'] != 191150] if row['valid'] else []
            view['top_alternative_annotation_overlaps'] = sorted(alternatives, key=lambda v: (-v['iou'], v['annotation_id']))[:5]
        selected.append(view)
    divergence = next((i for i, t in enumerate(record['token_ids'])
                       if str(i) not in cell['forced_actions'] and t != cell['expected_ids'][i]), None)
    return dict(rows=selected, alignment=alignment, malformed=result['malformed'], parser_status=result['parser_status'],
        burdens=result['burdens'], action_family_burdens=result['action_family_burdens'],
        first_free_divergence_action=divergence, stop_reason=record['stop_reason'],
        actual_free_actions=sum(str(i) not in cell['forced_actions'] for i in range(len(record['token_ids']))),
        censored_at_literal_budget=record['stop_reason'] == 'length')


def aligned_contrast(left, right, left_analysis, right_analysis):
    comparisons, unaligned = {}, []
    for channel in ('raw', 'median'):
        reference = {o['position']: o for o in left[f'{channel}_observations']}
        comparisons[channel] = []
        for o in right[f'{channel}_observations']:
            i = o['position']
            if i not in reference:
                continue
            role = right_analysis['alignment'].get(str(i))
            if role is None or role != left_analysis['alignment'].get(str(i)):
                if channel == 'raw':
                    unaligned.append(i)
                continue
            comparisons[channel].append(dict(position=i, semantic_role=role, **score_difference(reference[i], o)))
    return dict(score_comparisons=comparisons, unaligned_positions=unaligned)


def reuse_bottle(packet, fresh, fresh_analysis, tokenizer):
    old = a.load(packet['bindings']['B16-natural-351017']['path'])
    old_cell = next(c for c in packet['reuse_conditions'] if c['kind'] == 'natural')
    readout.validate_record(old, old_cell, packet)
    differences = []
    for channel in ('raw', 'median'):
        current = {o['position']: o for o in fresh[f'{channel}_observations']}
        for o in old[f'{channel}_observations']:
            if o['position'] not in current:
                continue
            now = current[o['position']]
            exact = all(o[k] == now[k] for k in ('coordinate_scores', 'full_log_normalizer', 'argmax',
                        'top5_noncoordinate', 'eos_score', 'object_start_score', 'prefix_token_ids'))
            differences.append(dict(channel=channel, position=o['position'], exact=exact, **score_difference(o, now)))
    compatible = bool(differences) and all(d['exact'] for d in differences)
    result = dict(status='compatible' if compatible else 'HOLD', natural_score_identity=differences,
        condition_identity='accepted source/runtime/checkpoint/prefix/processor bindings', contrasts=[])
    if not compatible:
        return result
    for old_cell in packet['reuse_conditions']:
        if old_cell['kind'] == 'natural':
            continue
        name = old_cell['condition'].replace('/', '-')
        record = a.load(packet['bindings'][name]['path'])
        readout.validate_record(record, old_cell, packet)
        sign = old_cell['edits'][0]['edited'] - old_cell['edits'][0]['original']
        cell = next(c for c in packet['conditions'] if c['image_id'] == 351017 and c['mode'] == 'free' and c['sign'] == sign)
        old_view = dict(cell, forced_actions={str(i): t for i, t in enumerate(old_cell['forced_prefix'])})
        analysis = analyze(record, old_view, packet, tokenizer)
        result['contrasts'].append(dict(sign=sign, clamped_artifact=packet['bindings'][name], record=record, analysis=analysis))
    return result


def consume(output, packet, tokenizer):
    """Saved rows and scores only; preserve HOLDs and semantic misalignment."""
    validate_conditions(packet)
    for identity in packet['bindings'].values():
        if a.digest(identity['path']) != identity['sha256']:
            raise ValueError('saved consumer input/reused artifact identity differs')
    output = Path(output)
    index = a.load(output / 'conditions.json')
    if [e['condition'] for e in index] != [c['condition'] for c in packet['conditions']]:
        raise ValueError('consumer condition index differs')
    records, analyses, rows, passed = {}, {}, [], {}
    for entry, cell in zip(index, packet['conditions'], strict=True):
        name, image = cell['condition'], cell['image_id']
        if entry['status'] == 'HOLD':
            rows.append(dict(condition=name, status='HOLD', reason=entry['reason']))
            continue
        if entry['status'] != 'completed' or (cell['mode'] != 'natural' and not passed.get(image)):
            raise ValueError('dependent artifact published after natural fidelity HOLD')
        path = output / entry['filename']
        if path.name != name.replace('/', '-') + '.json':
            raise ValueError('condition artifact filename differs')
        record = a.load(path)
        if record['schema'] != SCHEMA or record['condition'] != name or record['image_id'] != image:
            raise ValueError('record condition identity differs')
        validate_record(record, cell, packet)
        f = fidelity(record, cell) if cell['mode'] == 'natural' else None
        if f is not None:
            passed[image] = f['qualified']
        records[name] = record
        analyses[name] = analyze(record, cell, packet, tokenizer)
        rows.append(dict(condition=name, status='completed', artifact=binding(path), natural_fidelity=f,
            generated_actions=len(record['token_ids']), analysis=analyses[name],
            observation_metrics={channel: [dict(position=o['position'], semantic_role=analyses[name]['alignment'].get(str(o['position'])),
                **score_summary(o, record[f'{channel}_steps'][o['position']])) for o in record[f'{channel}_observations']]
                for channel in ('raw', 'median')}))
    if {p.name for p in output.glob('B16-*.json')} != {e['filename'] for e in index if e['status'] == 'completed'}:
        raise ValueError('unconsumed/extra condition artifact')
    contrasts, matched = [], []
    for cell in packet['conditions']:
        name = cell['condition']
        if cell['mode'] != 'free' or name not in records:
            continue
        reference = next(c['condition'] for c in packet['conditions'] if c['image_id'] == cell['image_id'] and c['mode'] == 'natural')
        contrasts.append(dict(condition=name, reference=reference,
            population='preceding-row-x2' if cell['image_id'] == 351017 else 'within-box-x1',
            pre_edit_score_identity={channel: [dict(position=o['position'],
                prefix_exact=o['prefix_token_ids'] == next(v for v in records[reference][f'{channel}_observations'] if v['position'] == o['position'])['prefix_token_ids'],
                **score_difference(next(v for v in records[reference][f'{channel}_observations'] if v['position'] == o['position']), o))
                for o in records[name][f'{channel}_observations'] if o['position'] <= cell['edit_position']]
                for channel in ('raw', 'median')},
            **aligned_contrast(records[reference], records[name], analyses[reference], analyses[name])))
        clamp = next((c['condition'] for c in packet['conditions'] if c['image_id'] == cell['image_id'] and
                      c['mode'] == 'clamped' and c['sign'] == cell['sign']), None)
        if clamp in records:
            matched.append(dict(free=name, clamped=clamp, restored='native y1 and x2 together',
                clamped_vs_natural=aligned_contrast(records[reference], records[clamp], analyses[reference], analyses[clamp]),
                **aligned_contrast(records[clamp], records[name], analyses[clamp], analyses[name])))
    bottle = next(c['condition'] for c in packet['conditions'] if c['image_id'] == 351017 and c['mode'] == 'natural')
    reuse = reuse_bottle(packet, records[bottle], analyses[bottle], tokenizer) if passed.get(351017) else dict(status='HOLD', reason='natural_fidelity')
    for reused in reuse.get('contrasts', []):
        name = next(c['condition'] for c in packet['conditions'] if c['image_id'] == 351017 and c['mode'] == 'free' and c['sign'] == reused['sign'])
        record = reused.pop('record')
        if name in records:
            matched.append(dict(free=name, clamped=reused['clamped_artifact'], restored='native actions17..28 (historical clamped score cell)',
                clamped_vs_natural=aligned_contrast(records[bottle], record, analyses[bottle], reused['analysis']),
                **aligned_contrast(record, records[name], reused['analysis'], analyses[name])))
    return dict(schema=SCHEMA, complete=all(e['status'] == 'completed' for e in index) and all(passed.values()) and reuse['status'] == 'compatible',
        conditions=rows, contrasts=contrasts, matched_free_clamped=matched, bottle_reuse=reuse,
        denominators=dict(within_box_x1_prescribed=4, preceding_row_x2_prescribed=2,
            within_box_x1_observed=sum(c['population'] == 'within-box-x1' for c in contrasts),
            preceding_row_x2_observed=sum(c['population'] == 'preceding-row-x2' for c in contrasts)),
        completed_requests=len(records), generated_actions=sum(len(r['token_ids']) for r in records.values()),
        limitations='Selected cached histories; no prevalence, physical-owner switch, unique training cause or training-objective comparison.')


def resource_excess(output, begin, *, native):
    values = readout.usage(output, native=native)
    return values, [k for k, v in values.items() if v > BOUNDS[k]] + (['wall_seconds'] if time.monotonic() - begin > 900 else [])


def run(config_path, output, *, cpu_factory=None):
    output = Path(output).resolve()
    if not output.is_relative_to(OUTPUT):
        raise ValueError('invocation output outside unit owner')
    output.mkdir(parents=True, exist_ok=False)
    begin, session, packet, statuses, passed = time.monotonic(), None, None, [], {}
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
            filename = cell['condition'].replace('/', '-') + '.json'
            a.write(output / filename, record)
            counts['completed_requests'] += 1
            counts['generated_actions'] += len(record['token_ids'])
            statuses.append(dict(condition=cell['condition'], status='completed', filename=filename))
            validate_record(record, cell, packet)
            if cell['mode'] == 'natural':
                f = fidelity(record, cell)
                passed[cell['image_id']] = f['qualified']
                a.write(output / f'fidelity-{cell["image_id"]}.json', f)
            a.write(output / f'resource-{counts["completed_requests"]:02d}.json',
                dict(seconds=time.monotonic() - begin, counts=counts, usage=resource_excess(output, begin, native=native)[0]))
        done = {e['condition'] for e in statuses}
        statuses.extend(dict(condition=c['condition'], status='HOLD', reason='resource_limit:' + ','.join(exceeded))
                        for c in packet['conditions'] if c['condition'] not in done)
        a.write(output / 'conditions.json', statuses)
        report = consume(output, packet, session.q.tokenizer)
        report['compute'] = 'native' if native else 'CPU_FIXTURE'
        a.write(output / 'readback.json', report)
        session.close()
        session = None
        check_payloads(packet['payloads'])
        # Verify exact entry binding at settlement without rehashing model payloads.
        load_packet(config_path, cpu=not native)
        if report['complete'] and not resource_excess(output, begin, native=native)[1]:
            status, code = 'complete', 0
    except Exception as error:
        a.write(output / 'error.json', dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc()))
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
    packet = a.load(config['input_packet']['path'])
    from probes import rollout_row_credit as retained
    q = retained.frontend()
    report = consume(args.output, packet, q.tokenizer)
    saved = a.load(args.output / 'readback.json')
    if report != {k: v for k, v in saved.items() if k != 'compute'}:
        raise ValueError('fresh saved consumer differs')
    print(json.dumps(dict(complete=report['complete'], completed_requests=report['completed_requests'],
        generated_actions=report['generated_actions'], denominators=report['denominators'])))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
