"""Four frozen first-row-history requests; no optimizer, replay, or extra forward."""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import resource
import subprocess
import time
import traceback
from pathlib import Path

from probes.rule_stability import artifacts as a, data

ROOT = a.ROOT
UNIT = ROOT / 'research/experiments/2026-10-03-first-row-history-cross'
OUTPUT = ROOT / 'outputs/research/physical-fn-recovery/2026-10-03/first-row-history-cross'
PREVIOUS = a.OUTPUT / 'native-primary-A-01'
SCHEMA = 'first-row-history-cross-v1'
IMAGE_ID = 351017
PERSON_ID = -1276804344338180
TARGET_ID = -4947389372712316
TARGET_BOX = [186, 30, 207, 106]
PREFIXES = dict(native=[151646, 8987, 151647, 151648, 151670, 151683, 152206, 152660, 151649],
                GT=[151646, 8987, 151647, 151648, 151670, 151694, 152187, 152669, 151649])
CONDITIONS = ['A0/native', 'A0/GT', 'A16/native', 'A16/GT']
BUDGET = 73


class PrefixForcer:
    """Record unforced median decisions, then force9 cached sequential actions."""

    def __init__(self, prefix, prompt_width):
        if list(prefix) not in PREFIXES.values() or type(prompt_width) is not int or prompt_width < 1:
            raise ValueError('unsupported frozen prefix or prompt width')
        self.prefix, self.prompt_width = tuple(prefix), prompt_width
        self.observations, self.calls = [], 0

    def __call__(self, input_ids, scores):
        import torch
        step = input_ids.shape[1] - self.prompt_width
        if input_ids.shape[0] != 1 or scores.shape[0] != 1 or step != self.calls:
            raise ValueError('forcing requires one sequential cached singleton history')
        self.calls += 1
        if step >= len(self.prefix):
            return scores
        token = self.prefix[step]
        argmax = int(scores[0].argmax())
        likelihood = float(torch.log_softmax(scores.float(), dim=-1)[0, token])
        if not math.isfinite(likelihood):
            raise ValueError('nonfinite pre-force median prefix likelihood')
        self.observations.append(dict(position=step, requested_token_id=token,
            pre_force_argmax=argmax, unforced_median_logprob=likelihood))
        forced = torch.full_like(scores, -torch.inf)
        forced[0, token] = 0
        return forced


def source_revision():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()


def small_binding(path):
    return dict(path=str(Path(path).resolve()), sha256=a.digest(path))


def image_inputs():
    images = a.load(data.FULL_LABEL_PATH)
    manifest = a.load(data.INPUT_MANIFEST_PATH)
    data.validate_inputs(images, manifest)
    image = next(image for image in images if image['image_id'] == IMAGE_ID)
    request = next(row for row in data.request_records(images, manifest) if row['image_id'] == IMAGE_ID)
    objects = sorted(image['objects'], key=lambda obj: (obj['bbox_2d'][0], obj['bbox_2d'][1], obj['coco_ann_id']))
    if (len(image['objects']) != 49 or objects[0]['coco_ann_id'] != PERSON_ID or
            objects[0]['bbox_2d'] != [0, 24, 517, 999] or objects[0]['desc'] != 'person' or
            objects[1]['coco_ann_id'] != TARGET_ID or objects[1]['bbox_2d'] != TARGET_BOX or
            objects[1]['desc'] != 'bottle'):
        raise ValueError('frozen selected person/next bottle identity changed')
    return image, request


def verify_frontend(q, image, request):
    from probes import iterative_positive as p
    encoded, sequence, owners = data.full_label_sequence(image, q)
    prompt = request['prompt_token_ids']
    if list(sequence.input_ids[:len(prompt)]) != prompt or owners[:2] != (f'full:{PERSON_ID}', f'full:{TARGET_ID}'):
        raise ValueError('maintained GT rendering changed prompt or owner order')
    if list(sequence.input_ids[len(prompt):len(prompt) + 9]) != PREFIXES['GT']:
        raise ValueError('maintained full-label first-row tokens differ')
    person = '<|object_ref_start|>person<|object_ref_end|><|box_start|>'
    for history, box in [('native', [0, 13, 536, 990]), ('GT', [0, 24, 517, 999])]:
        text = person + ''.join(f'<|coord_{x}|>' for x in box) + '<|box_end|>'
        if (q.tokenizer(text, add_special_tokens=False)['input_ids'] != PREFIXES[history] or
                q.tokenizer.decode(PREFIXES[history], skip_special_tokens=False) != text):
            raise ValueError('actual tokenizer first-row identity differs')
    batch = p.native_request(request, p.load(p.POLICY), q.processor)
    if (list(batch.prompt_token_ids[0]) != prompt or list(batch.image_grids[0]) != request['image_grid_thw'] or
            batch.media_sha256[0] != request['media_sha256']):
        raise ValueError('native prompt/media/grid identity differs')
    return batch


def prepare(directory):
    """Tokenizer/processor-only preparation; checkpoint JSON is not a tensor load."""
    from probes import rollout_row_credit as retained, iterative_positive as p
    from probes.rule_stability.__main__ import runtime_identity
    image, request = image_inputs()
    bindings = dict(labels=small_binding(data.FULL_LABEL_PATH), manifest=small_binding(data.INPUT_MANIFEST_PATH),
                    policy=small_binding(p.POLICY), image=small_binding(image['image_path']),
                    protocol=small_binding(UNIT / 'unit.md'), previous_invocation=small_binding(PREVIOUS / 'invocation.json'))
    if bindings['labels']['sha256'] != data.FULL_LABEL_SHA256 or bindings['manifest']['sha256'] != data.INPUT_MANIFEST_SHA256:
        raise ValueError('frozen full-label bytes/manifest changed')
    manifest = a.load(data.INPUT_MANIFEST_PATH)
    if manifest['sources'].get(str(p.POLICY)) != bindings['policy']['sha256'] or bindings['image']['sha256'] != image['image_sha256']:
        raise ValueError('original policy/image identity changed')
    q = retained.frontend()
    if q.model is not None:
        raise ValueError('preparation loaded a model')
    batch = verify_frontend(q, image, request)
    selected = {}
    for version in (0, 16):
        checkpoint = PREVIOUS / f'checkpoint-{version}'
        metadata = a.load(checkpoint / 'checkpoint.json')
        if any(metadata.get(k) != v for k, v in dict(schema=a.SCHEMA, arm='A', version=version, engine='native').items()):
            raise ValueError('selected checkpoint manifest identity differs')
        saved_path = PREVIOUS / f'rank-7/version-{version}/greedy-{IMAGE_ID}.json'
        saved = a.load(saved_path)
        if len(saved['token_ids']) <= BUDGET or saved['token_ids'][:9] != PREFIXES['native']:
            raise ValueError('saved native prefix or continuation length differs')
        if q.tokenizer.decode(saved['token_ids'], skip_special_tokens=False) != saved['text']:
            raise ValueError('selected saved token/text identity differs')
        selected[str(version)] = dict(checkpoint=str(checkpoint), manifest=small_binding(checkpoint / 'checkpoint.json'),
            saved=small_binding(saved_path), saved_action_count=len(saved['token_ids']), expected_ids=saved['token_ids'][:BUDGET])
    packet = dict(schema=SCHEMA, image=image, request=request, bindings=bindings, selected=selected,
        prefixes=PREFIXES, conditions=CONDITIONS, forced_actions=9, free_actions=64, budget=BUDGET,
        source_revision=source_revision(), runtime=runtime_identity(), model_loaded=False,
        prepared_prompt_tokens=len(batch.prompt_token_ids[0]), target=dict(coco_ann_id=TARGET_ID, bbox_2d=TARGET_BOX, desc='bottle'))
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    a.write(directory / 'input-packet.json', packet)
    proposal = dict(schema=SCHEMA, released=False, source_revision=source_revision(),
        runtime=packet['runtime'], input_packet=small_binding(directory / 'input-packet.json'),
        output=str(OUTPUT / 'native-01'), gpu=0, process_count=1,
        bounds=dict(requests=4, checkpoint_loads=2, actions=292, optimizer=0, backward=0, replay=0, exports=0),
        retry='no_automatic_relaunch')
    a.write(directory / 'native-proposal.json', proposal)
    return proposal


def load_packet(config_path, *, cpu=False):
    from probes.rule_stability.__main__ import runtime_identity
    config = a.load(config_path)
    if config.get('schema') != SCHEMA or config.get('bounds') != dict(requests=4, checkpoint_loads=2, actions=292,
            optimizer=0, backward=0, replay=0, exports=0) or config.get('gpu') != 0 or config.get('process_count') != 1:
        raise ValueError('frozen request/resource contract changed')
    identity = config['input_packet']
    if a.digest(identity['path']) != identity['sha256']:
        raise ValueError('input packet changed')
    packet = a.load(identity['path'])
    if (packet.get('schema') != SCHEMA or packet.get('prefixes') != PREFIXES or packet.get('conditions') != CONDITIONS or
            packet.get('forced_actions') != 9 or packet.get('free_actions') != 64 or packet.get('budget') != BUDGET or
            packet.get('model_loaded') is not False or config['runtime'] != runtime_identity()):
        raise ValueError('packet or runtime drift')
    if not cpu:
        if config.get('released') is not True or config['source_revision'] != source_revision():
            raise ValueError('native execution requires exact clean lead release')
        subprocess.run(['git', 'ls-files', '--error-unmatch', 'probes/first_row_history.py'], cwd=ROOT,
                       check=True, stdout=subprocess.DEVNULL)
        subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--', 'src', 'probes'], cwd=ROOT, check=True)
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
            raise ValueError('native release requires CUDA_VISIBLE_DEVICES=0')
    for binding in [*packet['bindings'].values(), *(row[key] for row in packet['selected'].values() for key in ('manifest', 'saved'))]:
        if a.digest(binding['path']) != binding['sha256']:
            raise ValueError(f"bound input changed: {binding['path']}")
    image, request = image_inputs()
    if image != packet['image'] or request != packet['request'] or packet['target'] != dict(coco_ann_id=TARGET_ID, bbox_2d=TARGET_BOX, desc='bottle'):
        raise ValueError('packet selected image/prompt/target drift')
    for version in (0, 16):
        selected = packet['selected'][str(version)]
        if (selected['checkpoint'] != str(PREVIOUS / f'checkpoint-{version}') or
                selected['saved']['path'] != str(PREVIOUS / f'rank-7/version-{version}/greedy-{IMAGE_ID}.json') or
                a.load(selected['saved']['path'])['token_ids'][:BUDGET] != selected['expected_ids']):
            raise ValueError('saved reference/checkpoint selection drift')
    return config, packet


def qualify_payload(packet):
    """Hash selected payloads once; reuse unchanged immutable-base evidence by stat."""
    from probes import iterative_positive as p
    policy = p.load(p.POLICY)
    previous = a.load(packet['bindings']['previous_invocation']['path'])['payload_qualification']
    evidence = {}
    for path, identity in previous.items():
        if Path(path).is_relative_to(Path(policy['base_model'])):
            stat = Path(path).stat()
            if (policy['payload_sha256'].get(path) != identity['sha256'] or
                    stat.st_size != identity['size_bytes'] or stat.st_mtime_ns != identity['mtime_ns']):
                raise ValueError(f'immutable base evidence cannot be reused: {path}')
            evidence[path] = dict(identity, qualification='reused original hash; unchanged size/mtime')
    expected_base = {path for path in policy['payload_sha256'] if Path(path).is_relative_to(Path(policy['base_model']))}
    if set(evidence) != expected_base or not evidence:
        raise ValueError('original immutable-base evidence is incomplete')
    for version in (0, 16):
        checkpoint = Path(packet['selected'][str(version)]['checkpoint'])
        meta = a.load(checkpoint / 'checkpoint.json')
        a.verify_manifest(checkpoint, meta['files'], excluded=('checkpoint.json',))
        for name, identity in meta['files'].items():
            path = checkpoint / name
            stat = path.stat()
            evidence[str(path)] = dict(identity, mtime_ns=stat.st_mtime_ns, qualification='fresh existing payload checker')
    return evidence


class NativeSession:
    def __init__(self, checkpoint, packet):
        from probes.rule_stability.runner import native_components
        from src.losses.vocab import build_token_vocabulary_groups
        from src.qwen.coordinate_policy import MedianPolicy
        self.q, self.delta, self.composition = native_components(checkpoint)
        self.q.model.eval()
        self.batch = verify_frontend(self.q, packet['image'], packet['request'])
        vocab = build_token_vocabulary_groups(self.q.token_identity, tokenizer=self.q.tokenizer)
        self.norm = MedianPolicy(self.q.model, list(vocab.coordinate))

    def generate(self, history):
        import torch
        from src.qwen.generation import generate_continuations, NativeGenerationPolicy
        forcer = PrefixForcer(PREFIXES[history], len(self.batch.prompt_token_ids[0]))
        begin = time.monotonic()
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            result = generate_continuations(self.q.model, self.batch, extensions=[()], budgets=[BUDGET],
                eos_token_id=self.q.tokenizer.convert_tokens_to_ids('<|im_end|>'), pad_token_id=self.q.tokenizer.pad_token_id,
                policy=NativeGenerationPolicy(temperature=0, top_p=1, top_k=0, repetition_penalty=1, use_model_defaults=False),
                trace='raw_and_policy', allow_pad_tokens=True,
                logits_processor=[self.norm.generation_transform(), forcer])[0]
        return dict(token_ids=list(result.token_ids), text=self.q.tokenizer.decode(result.token_ids, skip_special_tokens=False),
            stop_reason=result.stop_reason, raw_logprobs=list(result.raw_logprobs), policy_logprobs=list(result.policy_logprobs),
            prefix_observations=forcer.observations, processor_calls=forcer.calls, generation_seconds=time.monotonic() - begin)

    def close(self):
        import torch
        self.q = self.delta = self.norm = self.batch = None
        gc.collect()
        torch.cuda.empty_cache()


def validate_record(record, history):
    tokens = record['token_ids']
    if not 9 <= len(tokens) <= BUDGET or tokens[:9] != PREFIXES[history]:
        raise ValueError('literal forced prefix or action budget differs')
    if record['stop_reason'] not in ('im_end', 'length') or (record['stop_reason'] == 'length' and len(tokens) != BUDGET):
        raise ValueError('short continuation stop contract differs')
    if record['processor_calls'] != len(tokens) or len(record['prefix_observations']) != 9:
        raise ValueError('forcing processor skipped sequential actions')
    for position, observation in enumerate(record['prefix_observations']):
        if (observation['position'] != position or observation['requested_token_id'] != PREFIXES[history][position] or
                not math.isfinite(observation['unforced_median_logprob'])):
            raise ValueError('pre-force prefix evidence incomplete')
    for key in ('raw_logprobs', 'policy_logprobs'):
        if len(record[key]) != len(tokens) or any(not math.isfinite(x) for x in record[key]):
            raise ValueError('selected-action likelihood trace incomplete or nonfinite')


def natural_fidelity(record, expected):
    argmax_mismatches = [row for row in record['prefix_observations'] if row['pre_force_argmax'] != row['requested_token_id']]
    ids = record['token_ids']
    token_mismatches = [dict(position=i, expected=token, actual=ids[i] if i < len(ids) else None)
                        for i, token in enumerate(expected) if i >= len(ids) or ids[i] != token]
    qualified = not argmax_mismatches and not token_mismatches and len(ids) == BUDGET and record['stop_reason'] == 'length'
    return dict(qualified=qualified, pre_force_argmax_mismatches=argmax_mismatches,
                saved_first73_mismatches=token_mismatches, expected_stop='length')


def analyze(record, image, tokenizer):
    from probes.rule_stability.objectives import trajectory_analysis
    from src.eval.saved_rows import iou_xyxy
    analysis = trajectory_analysis(record, tokenizer)
    forced = [row for row in analysis['rows'] if row['positions'][0] < 9]
    if len(forced) != 1 or forced[0]['positions'] != list(range(9)) or not forced[0]['valid']:
        raise ValueError('forced first person is not exactly one complete valid9-action row')
    free = [dict(row) for row in analysis['rows'] if row['positions'][0] >= 9]
    for row in free:
        candidates = [dict(coco_ann_id=obj['coco_ann_id'], bbox_2d=obj['bbox_2d'], iou=iou_xyxy(row['bbox'], obj['bbox_2d']))
                      for obj in image['objects'] if obj['desc'] == row['description']] if row['valid'] else []
        row['same_category_owner_candidates'] = sorted(candidates, key=lambda obj: (-obj['iou'], obj['coco_ann_id']))
        row['best_same_category_iou'] = max((obj['iou'] for obj in candidates), default=None)
        row['target_bottle_iou'] = iou_xyxy(row['bbox'], TARGET_BOX) if row['valid'] and row['description'] == 'bottle' else None
        row['target_localized'] = row['target_bottle_iou'] is not None and row['target_bottle_iou'] >= .5
    first_bottle = next((row for row in free if row['description'] == 'bottle'), None)
    return dict(forced_row=forced[0], free_rows=free, first_free_category=free[0]['description'] if free else None,
        first_free_row=free[0] if free else None, first_bottle=first_bottle,
        any_free_target_localized=any(row['target_localized'] for row in free),
        duplicate_events_including_forced=analysis['duplicate_events'],
        invalid_free_rows=sum(not row['valid'] for row in free), malformed=analysis['malformed'],
        parser_status=analysis['parser_status'], free_actions=len(record['token_ids']) - 9,
        stop_reason=record['stop_reason'], budget_stop_is_local_censoring=True,
        matching_proxy='exact description and IoU>=.5 against annotation identities; supplied row excluded from coverage',
        duplicate_proxy='chronological strict class-agnostic IoU>.9, supplied row retained as an earlier partner')


def consume(output, packet, tokenizer):
    """Fresh reload of every published condition; no inference or reference rescoring."""
    output = Path(output)
    records = []
    for condition in CONDITIONS:
        path = output / (condition.replace('/', '-') + '.json')
        if not path.exists():
            break
        record = a.load(path)
        version, history = condition[1:].split('/')
        validate_record(record, history)
        fidelity = natural_fidelity(record, packet['selected'][version]['expected_ids']) if history == 'native' else None
        records.append(dict(condition=condition, artifact=small_binding(path), literal_text=record['text'], token_ids=record['token_ids'],
            natural_fidelity=fidelity, analysis=analyze(record, packet['image'], tokenizer)))
        if fidelity is not None and not fidelity['qualified']:
            break
    published = {path.name for path in output.glob('A*.json')}
    expected = {row['condition'].replace('/', '-') + '.json' for row in records}
    if published != expected:
        raise ValueError('condition artifacts violate HOLD ordering or contain unconsumed requests')
    return dict(schema=SCHEMA, complete=len(records) == 4 and all(row['natural_fidelity'] is None or
        row['natural_fidelity']['qualified'] for row in records), partial=len(records) != 4,
        conditions=records, completed_requests=len(records), generated_actions=sum(len(row['token_ids']) for row in records),
        limitations='Local annotation-relative conditional evidence only; no sustained native coverage or physical-FN recovery claim.')


def run(config_path, output, *, cpu_factory=None):
    """One process owns the exclusive output; CPU tests substitute computation only."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    begin = time.monotonic()
    counts = dict(checkpoint_loads=0, fixture_sessions=0, attempted_requests=0, completed_requests=0, generated_actions=0,
                  optimizer=0, backward=0, replay=0, exports=0)
    session = None
    status, code, evidence = 'failure', 1, {}
    a.write(output / 'invocation.json', dict(schema=SCHEMA, config=small_binding(config_path), pid=os.getpid(),
        started=time.time(), compute='CPU_FIXTURE' if cpu_factory else 'native', retry='no_automatic_relaunch'))
    try:
        config, packet = load_packet(config_path, cpu=cpu_factory is not None)
        if cpu_factory is None and str(output) != config['output']:
            raise ValueError('output differs from exact native release')
        qualification_begin = time.monotonic()
        evidence = qualify_payload(packet) if cpu_factory is None else {}
        a.write(output / 'qualification.json', dict(source_revision=config['source_revision'], runtime=config['runtime'],
            inputs=packet['bindings'], payloads=evidence, native_compute=cpu_factory is None,
            seconds=time.monotonic() - qualification_begin))
        for version in (0, 16):
            counts['checkpoint_loads' if cpu_factory is None else 'fixture_sessions'] += 1
            load_begin = time.monotonic()
            session = (cpu_factory or NativeSession)(Path(packet['selected'][str(version)]['checkpoint']), packet)
            a.write(output / f'load-A{version}.json', dict(seconds=time.monotonic() - load_begin, composition=session.composition))
            for history in ('native', 'GT'):
                counts['attempted_requests'] += 1
                record = session.generate(history)
                record.update(request_id=f'first-row-history:A{version}/{history}:{IMAGE_ID}', image_id=IMAGE_ID,
                    width=packet['image']['width'], height=packet['image']['height'], condition=f'A{version}/{history}',
                    likelihood_meaning=dict(prefix_policy='post-force selection, not ordinary policy likelihood',
                        prefix_unforced_median='prefix_observations', raw='ordinary raw selected-action likelihood',
                        free_policy='ordinary median-policy selected-action likelihood'))
                # Save literal evidence before technical validation so failure never conceals the result.
                a.write(output / f'A{version}-{history}.json', record)
                counts['completed_requests'] += 1
                counts['generated_actions'] += len(record['token_ids'])
                validate_record(record, history)
                if history == 'native':
                    fidelity = natural_fidelity(record, packet['selected'][str(version)]['expected_ids'])
                    a.write(output / f'fidelity-A{version}.json', fidelity)
                    if not fidelity['qualified']:
                        status, code = 'technical_HOLD', 2
                        break
            tokenizer = session.q.tokenizer
            session.close()
            session = None
            if code == 2:
                break
        readback_begin = time.monotonic()
        readback = consume(output, packet, tokenizer)
        readback['compute'] = 'CPU_FIXTURE' if cpu_factory else 'native'
        a.write(output / 'readback.json', readback)
        if code != 2:
            status, code = 'complete', 0
        for path, identity in evidence.items():
            stat = Path(path).stat()
            if stat.st_size != identity['size_bytes'] or stat.st_mtime_ns != identity['mtime_ns']:
                raise ValueError(f'qualified payload changed during invocation: {path}')
    except Exception as error:
        status, code = 'failure', 1
        a.write(output / 'error.json', dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc()))
    finally:
        if session is not None:
            try:
                session.close()
            except Exception as error:
                status, code = 'failure', 1
                a.write(output / 'cleanup-error.json', dict(type=type(error).__name__, message=str(error)))
        memory = dict(peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        if cpu_factory is None:
            import torch
            if torch.cuda.is_initialized():
                memory.update(peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                              peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
        a.write(output / 'terminal.json', dict(status=status, exit_code=code, counts=counts,
            seconds=time.monotonic() - begin, memory=memory, process_owner='single synchronous process; no child runtime',
            cleanup='session references released; no background compute', readback=small_binding(output / 'readback.json')
                if (output / 'readback.json').exists() else None))
    return code


def main(argv=None, *, cpu_factory=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'run', 'readback'])
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        prepare(args.output)
        return 0
    if args.command == 'run':
        if args.config is None:
            parser.error('run requires --config')
        return run(args.config, args.output, cpu_factory=cpu_factory)
    from probes import rollout_row_credit as retained
    invocation = a.load(args.output / 'invocation.json')
    terminal = a.load(args.output / 'terminal.json')
    if a.digest(invocation['config']['path']) != invocation['config']['sha256']:
        raise ValueError('invocation config changed')
    config = a.load(invocation['config']['path'])
    if a.digest(config['input_packet']['path']) != config['input_packet']['sha256']:
        raise ValueError('input packet changed')
    packet = a.load(config['input_packet']['path'])
    result = consume(args.output, packet, retained.frontend().tokenizer)
    if terminal['readback'] is None or a.digest(terminal['readback']['path']) != terminal['readback']['sha256']:
        raise ValueError('terminal readback identity changed')
    saved = a.load(terminal['readback']['path'])
    if dict(result, compute=invocation['compute']) != saved:
        raise ValueError('fresh consumer differs from published readback')
    print(json.dumps(dict(status=terminal['status'], exit_code=terminal['exit_code'], compute=invocation['compute'],
                         completed_requests=result['completed_requests'], generated_actions=result['generated_actions']), sort_keys=True))
    return terminal['exit_code']


if __name__ == '__main__':
    raise SystemExit(main())
