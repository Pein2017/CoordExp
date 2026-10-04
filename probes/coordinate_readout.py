"""Frozen selected-row geometry and sixteen cached coordinate readout requests."""
from __future__ import annotations

import argparse
import gc
import hashlib
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
UNIT = ROOT / 'research/experiments/2026-10-04-coordinate-readout-audit'
OUTPUT = ROOT / 'outputs/research/physical-fn-recovery/2026-10-04/coordinate-readout-audit'
PREVIOUS = a.OUTPUT / 'native-primary-B-01'
SCHEMA = 'coordinate-readout-audit-v1'
COORD_START = 151670
COORD_IDS = list(range(COORD_START, COORD_START + 1000))
RANKS = {351017: 7, 7511: 4, 13348: 3}
BOUNDS = dict(requests=16, actions=3226, checkpoint_loads=2, gpu=0, process_count=1,
    maximum_actions_per_request=440, maximum_context_tokens=1760, optimizer=0, backward=0,
    replay=0, training=0, warmup=0, exports=0, wall_seconds=900, rss_bytes=32 * 1024**3,
    cuda_allocated_bytes=12 * 1024**3, retained_bytes=32 * 1024**2)
SOURCE_PATHS = ['probes/coordinate_readout.py', 'tests/probes/test_coordinate_readout.py']


def binding(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=a.digest(path))


def revision():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()


def producer_identity():
    diff = subprocess.check_output(['git', 'diff', 'HEAD', '--', 'src', 'probes', *SOURCE_PATHS], cwd=ROOT)
    return dict(commit=revision(), diff_sha256=hashlib.sha256(diff).hexdigest(),
                files={path: a.digest(ROOT / path) for path in SOURCE_PATHS})


def definitions():
    """The sole frozen matrix: zero-based actions, prefix excludes the last action."""
    obs16 = {351017: sorted([*range(14, 18), *range(24, 28), *range(34, 38), *range(44, 48), 9, 19, 29, 39, 49]),
             7511: [*range(410, 414), *range(419, 423), *range(437, 440)],
             13348: [*range(231, 235), *range(240, 244)]}
    obs0 = {351017: [9, 18, 28, 39, *range(44, 48), 49],
            7511: [*range(31, 35), *range(40, 44)], 13348: [*range(50, 54), 55]}
    budget16, budget0 = {351017: 50, 7511: 440, 13348: 244}, {351017: 50, 7511: 45, 13348: 56}
    rows = []
    def add(version, image, name, budget, observations, edits=(), history_version=None):
        rows.append(dict(condition=f'B{version}/{name}/{image}', version=version, image_id=image,
            kind=name, history_version=version if history_version is None else history_version,
            budget=budget, observations=sorted(observations),
            edits=[dict(position=p, original=COORD_START + old, edited=COORD_START + new) for p, old, new in edits]))
    for image in RANKS:
        add(16, image, 'natural', budget16[image], obs16[image])
    for sign in (-1, 1):
        add(16, 7511, f'zero-width-{sign:+d}', 422, [419, 421], [(419, 684, 684 + sign)])
    for sign in (-1, 1):
        add(16, 13348, f'normal-{sign:+d}', 234, [231, 233], [(231, 544, 544 + sign)])
    for sign in (-1, 1):
        add(16, 351017, f'copy-{sign:+d}', 30, [16, 19, *range(24, 28), 29], [(16, 23, 23 + sign)])
    add(16, 13348, 'trusted-row', 235, [231, 232, 233, 234], [(231, 544, 546), (232, 632, 633), (233, 559, 556)])
    for image in RANKS:
        add(0, image, 'natural', budget0[image], obs0[image])
    for image in RANKS:
        add(0, image, 'B16-history', budget16[image], obs16[image], history_version=16)
    assert len(rows) == BOUNDS['requests'] and sum(r['budget'] for r in rows) == BOUNDS['actions']
    return rows


def saved_path(version, image, *, analysis=False):
    suffix = '-analysis' if analysis else ''
    return PREVIOUS / f'rank-{RANKS[image]}/version-{version}/greedy-{image}{suffix}.json'


def selectors(analysis, observations):
    positions = {}
    for row in analysis['rows']:
        for slot, position in enumerate(row['coordinate_positions']):
            if position in observations:
                positions[str(position)] = dict(role=('x1', 'y1', 'x2', 'y2')[slot], slot=slot,
                    row_order=row['order'], coordinate_positions=row['coordinate_positions'],
                    original_bbox=row.get('bbox'), original_valid=row['valid'], description=row['description'])
    for position in observations:
        positions.setdefault(str(position), dict(role='row_boundary', slot=None))
    return positions


def verify_selected_rows(saved, analysis, version, image):
    expected = {(16, 351017): [(1, [14, 15, 16, 17], [0, 0, 23, 86]),
                               (2, [24, 25, 26, 27], [0, 0, 23, 86])],
                (16, 7511): [(45, [410, 411, 412, 413], [670, 577, 677, 597]),
                              (46, [419, 420, 421, 422], [684, 577, 684, 597])],
                (16, 13348): [(25, [231, 232, 233, 234], [544, 632, 559, 682]),
                               (26, [240, 241, 242, 243], [584, 629, 599, 627])],
                (0, 7511): [(3, [31, 32, 33, 34], [44, 539, 53, 553]),
                             (4, [40, 41, 42, 43], [68, 538, 79, 553])],
                (0, 13348): [(5, [50, 51, 52, 53], [545, 632, 558, 685])]}
    for order, positions, bbox in expected.get((version, image), []):
        matches = [r for r in analysis['rows'] if r['order'] == order]
        if len(matches) != 1 or matches[0]['coordinate_positions'] != positions or matches[0]['bbox'] != bbox:
            raise ValueError('frozen saved row selector differs')
        if [saved['token_ids'][p] - COORD_START for p in positions] != bbox:
            raise ValueError('coordinate selector/token alignment differs')
    if (version, image) == (16, 7511) and [saved['token_ids'][p] - COORD_START for p in (437, 438, 439)] != [684, 577, 677]:
        raise ValueError('reversed-x selected prefix differs')


def verify_frontend(q, image, request):
    from probes import iterative_positive as p
    batch = p.native_request(request, p.load(p.POLICY), q.processor)
    if (list(batch.prompt_token_ids[0]) != request['prompt_token_ids'] or
            list(batch.image_grids[0]) != request['image_grid_thw'] or batch.media_sha256[0] != request['media_sha256']):
        raise ValueError('native prompt/media/grid identity differs')
    if [q.tokenizer.convert_tokens_to_ids(f'<|coord_{i}|>') for i in range(1000)] != COORD_IDS:
        raise ValueError('coordinate tokenizer ordering differs')
    return batch


def qualify_payload():
    """Reuse unchanged base receipts; hash only selected inference checkpoint files."""
    from probes import iterative_positive as p
    policy = p.load(p.POLICY)
    previous = a.load(PREVIOUS / 'invocation.json')['payload_qualification']
    evidence = {}
    expected_base = {path for path in policy['payload_sha256'] if Path(path).is_relative_to(Path(policy['base_model']))}
    for path in expected_base:
        identity, stat = previous[path], Path(path).stat()
        if (policy['payload_sha256'][path] != identity['sha256'] or stat.st_size != identity['size_bytes'] or
                stat.st_mtime_ns != identity['mtime_ns']):
            raise ValueError('immutable base receipt cannot be reused')
        evidence[path] = dict(identity, qualification='original hash, unchanged size/mtime')
    for version in (0, 16):
        checkpoint = PREVIOUS / f'checkpoint-{version}'
        meta = a.load(checkpoint / 'checkpoint.json')
        if any(meta.get(k) != v for k, v in dict(schema=a.SCHEMA, arm='B', version=version, engine='native').items()):
            raise ValueError('checkpoint endpoint identity differs')
        for name, identity in meta['files'].items():
            if not name.startswith(('adapter/', 'special_token_embeddings/')):
                continue
            path, stat = checkpoint / name, (checkpoint / name).stat()
            if stat.st_size != identity['size_bytes'] or a.digest(path) != identity['sha256']:
                raise ValueError(f'checkpoint inference payload differs: {path}')
            evidence[str(path)] = dict(identity, mtime_ns=stat.st_mtime_ns, qualification='fresh selected inference payload')
    return evidence


def distribution(values):
    import torch
    values = values.double()
    return dict(count=values.numel(), min=float(values.min()), median=float(values.median()),
        max=float(values.max()), mean=float(values.mean()),
        quantiles={str(q): float(torch.quantile(values, q)) for q in (.01, .1, .9, .99)})


def geometry(rows, pairs):
    import torch
    norms = torch.linalg.vector_norm(rows.double(), dim=1)
    if not bool(torch.isfinite(norms).all()) or bool((norms <= 0).any()):
        raise ValueError('nonfinite/zero selected embedding rows')
    unit = rows.double() / norms[:, None]
    groups = {}
    for i, row in enumerate(rows.contiguous()):
        groups.setdefault(row.view(torch.uint8).numpy().tobytes(), []).append(i)
    def pair(i, j):
        return dict(values=[i, j], cosine=float((unit[i] * unit[j]).sum()),
                    distance=float(torch.linalg.vector_norm(rows[i].double() - rows[j].double())))
    return dict(norms=distribution(norms), adjacent_cosine=distribution((unit[:-1] * unit[1:]).sum(1)),
        adjacent_distance=distribution(torch.linalg.vector_norm(rows[1:].double() - rows[:-1].double(), dim=1)),
        exact_duplicate_groups=[v for v in groups.values() if len(v) > 1], named_pairs=[pair(i, j) for i, j in pairs])


def static_geometry(directory, packet, evidence):
    import torch
    from safetensors import safe_open
    from probes import iterative_positive as p
    base = Path(p.load(p.POLICY)['base_model'])
    index = a.load(base / 'model.safetensors.index.json')['weight_map']
    key = 'model.language_model.embed_tokens.weight'
    with safe_open(str(base / index[key]), framework='pt', device='cpu') as handle:
        selected = handle.get_slice(key)
        if selected.get_shape() != [152670, 2048]:
            raise ValueError('base selected-row layout differs')
        base_rows = selected[COORD_START:COORD_START + 1000].to(torch.bfloat16)
    observed = {120, 121}
    for condition in packet['conditions']:
        for position in condition['observations']:
            token = condition['expected_ids'][position]
            if token in COORD_IDS:
                observed.add(token - COORD_START)
        observed.update(e['edited'] - COORD_START for e in condition['edits'])
    pairs = sorted({(v, n) if v < n else (n, v) for v in observed for n in (v - 1, v + 1) if 0 <= n < 1000})
    result, tensors = {}, {}
    for version in (0, 16):
        checkpoint = PREVIOUS / f'checkpoint-{version}'
        metadata = a.load(checkpoint / 'special_token_embeddings/special_token_embeddings.json')
        if (metadata['token_ids'][4:] != COORD_IDS or metadata['tensor_shape'] != [1004, 2048] or
                metadata['tensor_dtype'] != 'float32' or metadata['tie_word_embeddings'] is not False):
            raise ValueError('independent selected delta layout differs')
        with safe_open(str(checkpoint / 'special_token_embeddings/special_token_embeddings.safetensors'), framework='pt', device='cpu') as handle:
            if set(handle.keys()) != {'input_embed_delta', 'output_embed_delta'}:
                raise ValueError('independent delta tensor names differ')
            inp, out = [handle.get_slice(key)[4:1004] for key in ('input_embed_delta', 'output_embed_delta')]
        if any(t.dtype != torch.float32 or list(t.shape) != [1000, 2048] or not bool(torch.isfinite(t).all()) for t in (inp, out)):
            raise ValueError('selected delta dtype/shape/nonfinite')
        effective = dict(input_analytic=base_rows.float() + inp,
                         input_runtime=base_rows + inp.to(torch.bfloat16), output_analytic=base_rows.float() + out)
        norms = torch.linalg.vector_norm(effective['output_analytic'].double(), dim=1)
        factors = norms.median() / norms
        effective['output_normalized'] = effective['output_analytic'].double() * factors[:, None]
        tensors[version] = effective
        result[str(version)] = dict(rows={k: geometry(v, pairs) for k, v in effective.items()},
            output_norms=norms.tolist(), median=float(norms.median()), median_factors=factors.tolist(),
            analytic_vs_runtime_input=distribution(torch.linalg.vector_norm(
                effective['input_analytic'].double() - effective['input_runtime'].double(), dim=1)))
    changes = {kind: dict(distance=distribution(torch.linalg.vector_norm(
        tensors[16][kind].double() - tensors[0][kind].double(), dim=1)),
        cosine=distribution(torch.nn.functional.cosine_similarity(tensors[0][kind].double(), tensors[16][kind].double(), dim=1)),
        exact_equal_rows=[i for i in range(1000) if torch.equal(tensors[0][kind][i], tensors[16][kind][i])]) for kind in tensors[0]}
    report = dict(schema=SCHEMA, compute='CPU_SELECTED_ROWS', coordinate_ids=COORD_IDS, versions=result,
        same_coordinate_changes=changes, payloads=evidence, model_loaded=False, optimizer_loaded=False,
        selected_base_rows=1000, selected_delta_rows_per_tensor=1000,
        arithmetic=dict(input_analytic='BF16 base promoted FP32 + FP32 delta',
            input_runtime='BF16(base + BF16(delta))', output_analytic='BF16 base promoted FP32 + FP32 output delta',
            median='FP64 norms and lower median, FP64 factors',
            output_limitation='Actual logits separately accumulate and round base and delta terms; rows are algebraic descriptions.'))
    a.write(Path(directory) / 'static-geometry.json', report)
    return report


def prepare(directory):
    from probes import rollout_row_credit as retained, iterative_positive as p
    from probes.rule_stability.__main__ import runtime_identity
    if not (UNIT / 'unit.md').is_file():
        raise ValueError('authoritative unit is unavailable')
    images, manifest = data.load_inputs()
    requests = {r['image_id']: r for r in data.request_records(images, manifest)}
    images = {i['image_id']: i for i in images if i['image_id'] in RANKS}
    q = retained.frontend()
    if q.model is not None:
        raise ValueError('CPU preparation loaded a model')
    for image in RANKS:
        verify_frontend(q, images[image], requests[image])
    bindings = dict(protocol=binding(UNIT / 'unit.md'), labels=binding(data.FULL_LABEL_PATH),
        manifest=binding(data.INPUT_MANIFEST_PATH), policy=binding(p.POLICY), previous_invocation=binding(PREVIOUS / 'invocation.json'),
        previous_release=binding(a.OUTPUT / 'primary-release-01/primary-B-release.json'))
    saved, analyses = {}, {}
    for version in (0, 16):
        for image in RANKS:
            key = f'{version}/{image}'
            rawpath, analysispath = saved_path(version, image), saved_path(version, image, analysis=True)
            saved[key], analyses[key] = a.load(rawpath), a.load(analysispath)
            if q.tokenizer.decode(saved[key]['token_ids'], skip_special_tokens=False) != saved[key]['text']:
                raise ValueError('saved token/text identity differs')
            if saved[key]['prompt_token_ids'] != requests[image]['prompt_token_ids']:
                raise ValueError('saved prompt identity differs')
            verify_selected_rows(saved[key], analyses[key], version, image)
            bindings[f'saved-{key}'], bindings[f'analysis-{key}'] = binding(rawpath), binding(analysispath)
        bindings[f'checkpoint-{version}'] = binding(PREVIOUS / f'checkpoint-{version}/checkpoint.json')
    for image in RANKS:
        bindings[f'image-{image}'] = binding(images[image]['image_path'])
    conditions = definitions()
    for condition in conditions:
        key = f"{condition['history_version']}/{condition['image_id']}"
        expected = saved[key]['token_ids'][:condition['budget']]
        if len(expected) != condition['budget'] or len(requests[condition['image_id']]['prompt_token_ids']) + len(expected) > BOUNDS['maximum_context_tokens']:
            raise ValueError('saved history/context budget differs')
        prefix = expected[:-1].copy()
        for edit in condition['edits']:
            if prefix[edit['position']] != edit['original']:
                raise ValueError('intervention original token differs')
            prefix[edit['position']] = edit['edited']
        condition.update(expected_ids=expected, forced_prefix=prefix, selectors=selectors(analyses[key], condition['observations']),
            saved_raw_logprobs=saved[key]['raw_logprobs'][:condition['budget']],
            saved_policy_logprobs=saved[key]['policy_logprobs'][:condition['budget']])
    worker = next(r for r in analyses['16/13348']['rows'] if r['order'] == 25)
    owners = [o for o in images[13348]['objects'] if o['coco_ann_id'] == 191150]
    if (not worker['valid'] or worker['description'] != 'person' or len(owners) != 1 or
            owners[0]['desc'] != 'person' or owners[0]['bbox_2d'] != [546, 633, 556, 682]):
        raise ValueError('trusted normal-worker owner binding is ambiguous')
    packet = dict(schema=SCHEMA, conditions=conditions, images={str(k): v for k, v in images.items()},
        requests={str(k): requests[k] for k in RANKS}, bindings=bindings, bounds=BOUNDS,
        runtime=runtime_identity(), producer=producer_identity(), model_loaded=False,
        coordinate_ids=COORD_IDS, eos_id=q.tokenizer.convert_tokens_to_ids('<|im_end|>'),
        object_start_id=q.tokenizer.convert_tokens_to_ids('<|object_ref_start|>'),
        trusted_owner=dict(ann_id=191150, row_order=25, bbox=[546, 633, 556, 682],
                           binding='root-prescribed trusted normal-worker control; exact annotation/category/box'))
    evidence = qualify_payload()
    directory = Path(directory).resolve()
    if not directory.is_relative_to(OUTPUT):
        raise ValueError('preparation output is outside unit owner')
    directory.mkdir(parents=True, exist_ok=False)
    static_geometry(directory, packet, evidence)
    packet['static_geometry'] = binding(directory / 'static-geometry.json')
    packet['payloads'] = evidence
    a.write(directory / 'input-packet.json', packet)
    proposal = dict(schema=SCHEMA, released=False, source_revision=revision(), producer_files=packet['producer']['files'],
        input_packet=binding(directory / 'input-packet.json'), runtime=packet['runtime'], bounds=BOUNDS,
        output=str(OUTPUT / 'native-01'), retry='no_automatic_relaunch')
    a.write(directory / 'native-proposal.json', proposal)
    return proposal


def validate_conditions(packet):
    if packet.get('schema') != SCHEMA or packet.get('bounds') != BOUNDS or packet.get('coordinate_ids') != COORD_IDS:
        raise ValueError('frozen packet/resources differ')
    rows = packet['conditions']
    if len(rows) != 16:
        raise ValueError('frozen condition count differs')
    for row, declared in zip(rows, definitions(), strict=True):
        if any(row.get(k) != v for k, v in declared.items()):
            raise ValueError('frozen condition selector/edit/budget differs')
        prefix = row['expected_ids'][:-1].copy()
        for edit in row['edits']:
            if prefix[edit['position']] != edit['original']:
                raise ValueError('frozen original action differs')
            prefix[edit['position']] = edit['edited']
        if row['forced_prefix'] != prefix or len(prefix) != row['budget'] - 1 or set(row['selectors']) != {str(p) for p in row['observations']}:
            raise ValueError('literal prefix/action alignment differs')


def load_packet(config_path, *, cpu=False):
    from probes.rule_stability.__main__ import runtime_identity
    config = a.load(config_path)
    if config.get('schema') != SCHEMA or config.get('bounds') != BOUNDS or config.get('retry') != 'no_automatic_relaunch':
        raise ValueError('frozen release/resource contract differs')
    if a.digest(config['input_packet']['path']) != config['input_packet']['sha256']:
        raise ValueError('input packet identity differs')
    packet = a.load(config['input_packet']['path'])
    validate_conditions(packet)
    if packet['model_loaded'] is not False or config['runtime'] != runtime_identity() or config['runtime'] != packet['runtime']:
        raise ValueError('effective runtime/CPU preparation identity differs')
    if config['producer_files'] != packet['producer']['files'] or config['producer_files'] != producer_identity()['files']:
        raise ValueError('qualified producer bytes differ')
    if not cpu:
        if config.get('released') is not True or config.get('source_revision') != revision():
            raise ValueError('exact clean lead release is required')
        if subprocess.check_output(['git', 'status', '--porcelain=v1', '--untracked-files=all'], cwd=ROOT):
            raise ValueError('native producer checkout is dirty')
        subprocess.run(['git', 'ls-files', '--error-unmatch', *SOURCE_PATHS], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
            raise ValueError('one serial GPU requires CUDA_VISIBLE_DEVICES=0')
    for item in [*packet['bindings'].values(), packet['static_geometry']]:
        if a.digest(item['path']) != item['sha256']:
            raise ValueError(f"bound input differs: {item['path']}")
    for path, identity in packet['payloads'].items():
        stat = Path(path).stat()
        if stat.st_size != identity['size_bytes'] or stat.st_mtime_ns != identity['mtime_ns']:
            raise ValueError(f'qualified payload differs: {path}')
    for row in packet['conditions']:
        raw = a.load(saved_path(row['history_version'], row['image_id']))
        analysis = a.load(saved_path(row['history_version'], row['image_id'], analysis=True))
        if raw['token_ids'][:row['budget']] != row['expected_ids'] or selectors(analysis, row['observations']) != row['selectors']:
            raise ValueError('saved-reference/selector binding differs')
    return config, packet


def capture_scores(scores, packet):
    import torch
    values = scores[0].float()
    if scores.shape[0] != 1 or not bool(torch.isfinite(values).all()):
        raise ValueError('nonfinite or nonsingleton incoming scores')
    noncoords = values.clone()
    noncoords[COORD_START:COORD_START + 1000] = -torch.inf
    top = torch.topk(noncoords, 5)
    return dict(coordinate_scores=values[COORD_START:COORD_START + 1000].cpu().tolist(),
        full_log_normalizer=float(torch.logsumexp(values, -1)), argmax=int(values.argmax()),
        top5_noncoordinate=[dict(token_id=int(i), score=float(s)) for i, s in zip(top.indices, top.values)],
        eos_score=float(values[packet['eos_id']]), object_start_score=float(values[packet['object_start_id']]))


class ReadoutProcessor:
    """Observe the real incoming channel, then optionally force a fresh tensor."""
    def __init__(self, condition, packet, prompt_width, *, force):
        self.condition, self.packet, self.prompt_width, self.force = condition, packet, prompt_width, force
        self.calls, self.steps, self.observations = 0, [], []

    def __call__(self, input_ids, scores):
        import torch
        position = input_ids.shape[1] - self.prompt_width
        if input_ids.shape[0] != 1 or scores.shape[0] != 1 or position != self.calls or position >= self.condition['budget']:
            raise ValueError('readout requires sequential cached singleton actions')
        history = input_ids[0, self.prompt_width:].tolist()
        if history != self.condition['forced_prefix'][:position]:
            raise ValueError('actual cached prefix differs from declared literal history')
        self.calls += 1
        argmax = int(scores[0].argmax())
        requested = self.condition['forced_prefix'][position] if position < len(self.condition['forced_prefix']) else argmax
        logprob = float(torch.log_softmax(scores.float(), -1)[0, requested])
        if not math.isfinite(logprob):
            raise ValueError('nonfinite pre-force selected likelihood')
        self.steps.append(dict(position=position, argmax=argmax, requested_token_id=requested, pre_force_logprob=logprob))
        if position in self.condition['observations']:
            self.observations.append(dict(position=position, prefix_token_ids=history,
                selector=self.condition['selectors'][str(position)], **capture_scores(scores, self.packet)))
        if self.force and position < len(self.condition['forced_prefix']):
            forced = torch.full_like(scores, -torch.inf)
            forced[0, requested] = 0
            return forced
        return scores


class NativeSession:
    def __init__(self, checkpoint, packet):
        from probes.rule_stability.runner import native_components
        from src.qwen.coordinate_policy import MedianPolicy
        self.q, self.delta, self.composition = native_components(checkpoint)
        self.q.model.eval().requires_grad_(False)
        self.norm = MedianPolicy(self.q.model, COORD_IDS)
        self.batches = {image: verify_frontend(self.q, packet['images'][image], request)
                        for image, request in packet['requests'].items()}
        self.packet = packet

    def generate(self, condition):
        import torch
        from src.qwen.generation import generate_continuations, NativeGenerationPolicy
        batch = self.batches[str(condition['image_id'])]
        width = len(batch.prompt_token_ids[0])
        raw = ReadoutProcessor(condition, self.packet, width, force=False)
        median = ReadoutProcessor(condition, self.packet, width, force=True)
        begin = time.monotonic()
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            result = generate_continuations(self.q.model, batch, extensions=[()], budgets=[condition['budget']],
                eos_token_id=self.packet['eos_id'], pad_token_id=self.q.tokenizer.pad_token_id,
                policy=NativeGenerationPolicy(temperature=0, top_p=1, top_k=0, repetition_penalty=1, use_model_defaults=False),
                trace='raw_and_policy', allow_pad_tokens=True,
                logits_processor=[raw, self.norm.generation_transform(), median])[0]
        # Raw and normalized winners can differ at the sole free action. Bind
        # its raw likelihood to the actual policy emission; retain raw argmax.
        raw.steps[-1]['requested_token_id'] = result.token_ids[-1]
        raw.steps[-1]['pre_force_logprob'] = result.raw_logprobs[-1]
        return dict(token_ids=list(result.token_ids), text=self.q.tokenizer.decode(result.token_ids, skip_special_tokens=False),
            stop_reason=result.stop_reason, raw_logprobs=list(result.raw_logprobs), policy_logprobs=list(result.policy_logprobs),
            raw_steps=raw.steps, median_steps=median.steps, raw_observations=raw.observations,
            median_observations=median.observations, generation_seconds=time.monotonic() - begin)

    def close(self):
        import torch
        self.q = self.delta = self.norm = self.batches = None
        gc.collect()
        torch.cuda.empty_cache()


def validate_record(record, condition, packet):
    tokens = record['token_ids']
    if len(tokens) != condition['budget'] or tokens[:-1] != condition['forced_prefix']:
        raise ValueError('literal forced history or action budget differs')
    expected_stop = 'im_end' if tokens[-1] == packet['eos_id'] else 'length'
    if record['stop_reason'] != expected_stop:
        raise ValueError('terminal action/stop reason differs')
    for channel in ('raw', 'median'):
        steps, observations = record[f'{channel}_steps'], record[f'{channel}_observations']
        if len(steps) != len(tokens) or [s['position'] for s in steps] != list(range(len(tokens))):
            raise ValueError('every-step pre-force evidence is incomplete')
        if [r['position'] for r in observations] != condition['observations']:
            raise ValueError('compact observation positions differ')
        for position, step in enumerate(steps):
            if step['requested_token_id'] != tokens[position] or not math.isfinite(step['pre_force_logprob']):
                raise ValueError('pre-force selected action differs')
        for row in observations:
            position = row['position']
            if row['prefix_token_ids'] != tokens[:position] or row['selector'] != condition['selectors'][str(position)]:
                raise ValueError('captured prefix/selector differs')
            if len(row['coordinate_scores']) != 1000 or len(row['top5_noncoordinate']) != 5:
                raise ValueError('compact vocabulary support differs')
            scalars = [*row['coordinate_scores'], row['full_log_normalizer'], row['eos_score'], row['object_start_score'],
                       *(r['score'] for r in row['top5_noncoordinate'])]
            if not all(math.isfinite(v) for v in scalars) or any(r['token_id'] in COORD_IDS for r in row['top5_noncoordinate']):
                raise ValueError('compact evidence is nonfinite or noncoordinate support differs')
            if row['argmax'] != steps[position]['argmax']:
                raise ValueError('capture/step argmax differs')
    for key in ('raw_logprobs', 'policy_logprobs'):
        if len(record[key]) != len(tokens) or not all(math.isfinite(v) for v in record[key]):
            raise ValueError('selected-action likelihood evidence differs')


def fidelity(record, condition):
    expected = condition['expected_ids']
    mismatch = [dict(position=i, expected=t, actual=record['token_ids'][i], argmax=record['median_steps'][i]['argmax'])
                for i, t in enumerate(expected) if record['token_ids'][i] != t or record['median_steps'][i]['argmax'] != t]
    return dict(qualified=not mismatch, mismatch_count=len(mismatch), mismatches=mismatch,
        raw_logprob_max_abs_difference=max(abs(a - b) for a, b in zip(record['raw_logprobs'], condition['saved_raw_logprobs'])),
        median_logprob_max_abs_difference=max(abs(s['pre_force_logprob'] - b)
            for s, b in zip(record['median_steps'], condition['saved_policy_logprobs'])),
        likelihood_differences_are_descriptive=True)


def observation_metrics(observation, emitted):
    import torch
    scores = torch.tensor(observation['coordinate_scores'], dtype=torch.float64)
    role, prefix = observation['selector']['role'], observation['prefix_token_ids']
    slot = observation['selector']['slot']
    lo, hi = (0, 999) if slot in (0, 1) else (None, None)
    if slot in (2, 3):
        predecessor = prefix[observation['position'] - 2] - COORD_START
        if 0 <= predecessor < 999:
            lo, hi = predecessor + 1, 1000
    best_legal = float(scores[lo:hi].max()) if lo is not None else None
    if best_legal is not None:
        illegal = torch.cat((scores[:lo], scores[hi:]))
        best_illegal = max(float(illegal.max()) if len(illegal) else -math.inf,
                           observation['top5_noncoordinate'][0]['score'])
    else:
        best_illegal = None
    value = emitted - COORD_START
    neighbors = {str(n): float(scores[value] - scores[n]) for n in (value - 1, value + 1) if 0 <= value < 1000 and 0 <= n < 1000}
    return dict(role=role, emitted_token_id=emitted, coordinate_argmax=int(scores.argmax()),
        coordinate_probability_mass=math.exp(float(torch.logsumexp(scores, -1)) - observation['full_log_normalizer']),
        emitted_minus_neighbor_scores=neighbors, legal_coordinate_interval=[lo, hi] if lo is not None else None,
        best_legal_minus_best_illegal=best_legal - best_illegal if best_legal is not None else None,
        illegal_support='all noncoordinate tokens plus causally illegal coordinate values',
        eos_logprob=observation['eos_score'] - observation['full_log_normalizer'],
        object_start_logprob=observation['object_start_score'] - observation['full_log_normalizer'])


def differences(left, right):
    import torch
    a_scores = torch.tensor(left['coordinate_scores'], dtype=torch.float64) - left['full_log_normalizer']
    b_scores = torch.tensor(right['coordinate_scores'], dtype=torch.float64) - right['full_log_normalizer']
    delta = b_scores - a_scores
    return dict(exact_equal=bool(torch.equal(a_scores, b_scores)), max_abs=float(delta.abs().max()),
        rms=float(torch.sqrt((delta * delta).mean())), full_argmax_changed=left['argmax'] != right['argmax'],
        coordinate_argmax_changed=int(a_scores.argmax()) != int(b_scores.argmax()))


def consume(output, packet):
    """Read saved evidence only; enumerate skipped cells and controlled contrasts."""
    output = Path(output)
    index = a.load(output / 'conditions.json')
    if [r['condition'] for r in index] != [r['condition'] for r in packet['conditions']]:
        raise ValueError('consumer condition index differs')
    records, rows, passed = {}, [], {}
    for entry, condition in zip(index, packet['conditions'], strict=True):
        dependencies = [(condition['version'], condition['image_id'])] if condition['kind'] != 'natural' else []
        if condition['kind'] == 'B16-history':
            dependencies.append((16, condition['image_id']))
        allowed = all(passed.get(key) for key in dependencies)
        if entry['status'] == 'completed':
            if not allowed:
                raise ValueError('dependent artifact published after natural fidelity HOLD')
            path = output / entry['filename']
            if path.name != condition['condition'].replace('/', '-') + '.json':
                raise ValueError('condition artifact filename differs')
            record = a.load(path)
            validate_record(record, condition, packet)
            natural = fidelity(record, condition) if condition['kind'] == 'natural' else None
            if natural is not None:
                passed[(condition['version'], condition['image_id'])] = natural['qualified']
            records[condition['condition']] = record
            row = dict(condition=condition['condition'], status='completed', artifact=binding(path),
                kind=condition['kind'], natural_fidelity=natural, generated_actions=len(record['token_ids']),
                literal_prefix=condition['forced_prefix'], final_token_id=record['token_ids'][-1],
                observation_metrics={channel: [dict(position=o['position'], **observation_metrics(o, record['token_ids'][o['position']]))
                    for o in record[f'{channel}_observations']] for channel in ('raw', 'median')})
        elif entry['status'] == 'HOLD':
            row = dict(condition=condition['condition'], status='HOLD', reason=entry['reason'], kind=condition['kind'])
        else:
            raise ValueError('consumer condition state differs')
        rows.append(row)
    published = {p.name for p in output.glob('B*.json')}
    if published != {e['filename'] for e in index if e['status'] == 'completed'}:
        raise ValueError('unconsumed/extra condition artifacts')
    contrasts = []
    for condition in packet['conditions']:
        name = condition['condition']
        if condition['kind'] == 'natural' or name not in records:
            continue
        natural_name = f"B16/natural/{condition['image_id']}"
        if natural_name not in records:
            raise ValueError('contrast natural reference unavailable')
        for channel in ('raw', 'median'):
            reference = {o['position']: o for o in records[natural_name][f'{channel}_observations']}
            for observed in records[name][f'{channel}_observations']:
                if observed['position'] in reference:
                    contrasts.append(dict(condition=name, reference=natural_name, channel=channel,
                        position=observed['position'], **differences(reference[observed['position']], observed)))
    complete = all(r['status'] == 'completed' and (r['natural_fidelity'] is None or r['natural_fidelity']['qualified']) for r in rows)
    return dict(schema=SCHEMA, complete=complete, conditions=rows, contrasts=contrasts,
        completed_requests=len(records), generated_actions=sum(len(r['token_ids']) for r in records.values()),
        static_geometry=packet['static_geometry'], coordinate_ids=COORD_IDS,
        limitations='Selected rows and supplied cached histories only; no natural-loop escape, unique causal diagnosis, physical-FN or training claim.')


def usage(output, *, native):
    values = dict(rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                  retained_bytes=sum(p.stat().st_size for p in Path(output).rglob('*') if p.is_file()))
    if native:
        import torch
        values['cuda_allocated_bytes'] = torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0
    return values


def resource_excess(values, elapsed):
    return [key for key, value in values.items() if value > BOUNDS[key]] + (['wall_seconds'] if elapsed > BOUNDS['wall_seconds'] else [])


def run(config_path, output, *, cpu_factory=None):
    output = Path(output).resolve()
    if not output.is_relative_to(OUTPUT):
        raise ValueError('invocation output is outside unit owner')
    output.mkdir(parents=True, exist_ok=False)
    begin, session, statuses, passed = time.monotonic(), None, [], {}
    counts = dict(checkpoint_loads=0, fixture_sessions=0, attempted_requests=0, completed_requests=0, generated_actions=0,
                  optimizer=0, backward=0, replay=0, training=0, warmup=0, exports=0)
    status, code, packet = 'technical_HOLD', 2, None
    native = cpu_factory is None
    a.write(output / 'invocation.json', dict(schema=SCHEMA, config=binding(config_path), pid=os.getpid(),
        started=time.time(), compute='native' if native else 'CPU_FIXTURE', retry='no_automatic_relaunch'))
    try:
        config, packet = load_packet(config_path, cpu=not native)
        if native and str(output) != config['output']:
            raise ValueError('output differs from exact release')
        a.write(output / 'qualification.json', dict(producer=config['producer_files'], source_revision=config['source_revision'],
            runtime=config['runtime'], inputs=packet['bindings'], payloads=packet['payloads'], native_compute=native))
        stop = []
        for version in (16, 0):
            stop = resource_excess(usage(output, native=native), time.monotonic() - begin)
            if stop:
                break
            counts['checkpoint_loads' if native else 'fixture_sessions'] += 1
            load_begin = time.monotonic()
            session = (cpu_factory or NativeSession)(PREVIOUS / f'checkpoint-{version}', packet)
            a.write(output / f'load-B{version}.json', dict(seconds=time.monotonic() - load_begin, composition=session.composition))
            for condition in [c for c in packet['conditions'] if c['version'] == version]:
                dependencies = [(version, condition['image_id'])] if condition['kind'] != 'natural' else []
                if condition['kind'] == 'B16-history':
                    dependencies.append((16, condition['image_id']))
                if not all(passed.get(key) for key in dependencies):
                    statuses.append(dict(condition=condition['condition'], status='HOLD', reason='natural_fidelity_dependency'))
                    continue
                stop = resource_excess(usage(output, native=native), time.monotonic() - begin)
                if stop:
                    break
                counts['attempted_requests'] += 1
                record = session.generate(condition)
                record.update(schema=SCHEMA, condition=condition['condition'], image_id=condition['image_id'],
                    coordinate_ids=COORD_IDS, edits=condition['edits'], conditioning='cached sequential forced history; last action free',
                    likelihood_meaning=dict(raw='unforced raw selected-action', median='median_steps: pre-force selected-action',
                        policy='post-force selection; forced actions have likelihood zero'))
                filename = condition['condition'].replace('/', '-') + '.json'
                a.write(output / filename, record)
                counts['completed_requests'] += 1
                counts['generated_actions'] += len(record['token_ids'])
                statuses.append(dict(condition=condition['condition'], status='completed', filename=filename))
                validate_record(record, condition, packet)
                if condition['kind'] == 'natural':
                    result = fidelity(record, condition)
                    passed[(version, condition['image_id'])] = result['qualified']
                    a.write(output / f'fidelity-B{version}-{condition["image_id"]}.json', result)
            session.close()
            session = None
            if stop:
                break
        existing = {s['condition'] for s in statuses}
        statuses.extend(dict(condition=c['condition'], status='HOLD', reason='resource_limit:' + ','.join(stop))
                        for c in packet['conditions'] if c['condition'] not in existing)
        statuses.sort(key=lambda s: [c['condition'] for c in packet['conditions']].index(s['condition']))
        a.write(output / 'conditions.json', statuses)
        readback = consume(output, packet)
        readback['compute'] = 'native' if native else 'CPU_FIXTURE'
        a.write(output / 'readback.json', readback)
        exceeded = resource_excess(usage(output, native=native), time.monotonic() - begin)
        if readback['complete'] and not exceeded:
            status, code = 'complete', 0
        for path, identity in packet['payloads'].items():
            stat = Path(path).stat()
            if stat.st_size != identity['size_bytes'] or stat.st_mtime_ns != identity['mtime_ns']:
                raise ValueError('qualified payload changed during invocation')
    except Exception as error:
        a.write(output / 'error.json', dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc()))
        if packet is not None and not (output / 'conditions.json').exists():
            existing = {s['condition'] for s in statuses}
            statuses.extend(dict(condition=c['condition'], status='HOLD', reason=f'execution_failure:{type(error).__name__}')
                            for c in packet['conditions'] if c['condition'] not in existing)
            statuses.sort(key=lambda s: [c['condition'] for c in packet['conditions']].index(s['condition']))
            a.write(output / 'conditions.json', statuses)
    finally:
        if session is not None:
            try:
                session.close()
            except Exception as error:
                a.write(output / 'cleanup-error.json', dict(type=type(error).__name__, message=str(error)))
                status, code = 'technical_HOLD', 2
        current = usage(output, native=native)
        a.write(output / 'terminal.json', dict(schema=SCHEMA, status=status, exit_code=code, counts=counts,
            seconds=time.monotonic() - begin, usage=current,
            resource_excess=resource_excess(current, time.monotonic() - begin),
            process_owner='single synchronous process, no background compute', cleanup='session references released',
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
        print(json.dumps(dict(status='CPU_PREPARED', input_packet=result['input_packet'], model_loaded=False), sort_keys=True))
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
    validate_conditions(packet)
    report = consume(args.output, packet)
    saved = a.load(args.output / 'readback.json')
    if report != {k: v for k, v in saved.items() if k != 'compute'}:
        raise ValueError('fresh consumer differs from saved readback')
    print(json.dumps(dict(complete=report['complete'], completed_requests=report['completed_requests'],
        generated_actions=report['generated_actions']), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
