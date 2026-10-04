"""Frozen visual perturbation transport at two B16 pre-x1 residual states."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
import traceback
from pathlib import Path

from probes import box_continuity as continuity, coordinate_readout as readout
from probes.coordinate_diagnostics import owner_entry
from probes.rule_stability import artifacts as a

ROOT, PREVIOUS = a.ROOT, readout.PREVIOUS
UNIT = ROOT / 'research/experiments/2026-10-04-visual-state-localization'
OUTPUT = ROOT / 'outputs/research/physical-fn-recovery/2026-10-04/visual-state-localization'
SCHEMA = 'visual-state-localization-v1'
COORD_START, COORD_IDS = readout.COORD_START, readout.COORD_IDS
SOURCE_PATHS = ['probes/coordinate_diagnostics/visual_state.py',
                'tests/probes/coordinate_diagnostics/test_visual_state.py']
BOUNDS = dict(requests=20, actions=2550, checkpoint_loads=1, gpu=0, process_count=1,
    maximum_actions_per_request=236, maximum_context_tokens=1598, optimizer=0, backward=0,
    replay=0, training=0, warmup=0, exports=0, wall_seconds=900,
    rss_bytes=32 * 1024**3, cuda_allocated_bytes=12 * 1024**3, retained_bytes=64 * 1024**2)
SITES = {
    351017: dict(name='bottle', owner='target', position=14, budget=19, row_start=9,
        target=[232, 25, 258, 88], background=[1208, 0, 1234, 63], fill=[63, 37, 35],
        ring_counts=[1680, 1344], pixel_L1=[127190, 69999]),
    13348: dict(name='person', owner='person', position=231, budget=236, row_start=227,
        target=[681, 527, 694, 567], background=[742, 580, 755, 620], fill=[132, 124, 115],
        ring_counts=[1104, 1104], pixel_L1=[109446, 61071])}
BOUNDARIES = {'early': 2, 'middle': 13, 'final': 27}
binding, revision = readout.binding, readout.revision


def producer_identity():
    diff = subprocess.check_output(['git', 'diff', 'HEAD', '--', *SOURCE_PATHS], cwd=ROOT)
    return dict(commit=revision(), diff_sha256=hashlib.sha256(diff).hexdigest(),
                files={p: a.digest(ROOT / p) for p in SOURCE_PATHS})


def definitions():
    cells = []
    for phase, regions in [('clean', ['clean']), ('noop', ['clean']), ('image', ['target', 'background']),
                           ('final', ['target', 'background']), ('early', ['target', 'background']),
                           ('middle', ['target', 'background'])]:
        for image, s in SITES.items():
            for region in regions:
                name = f'{s["name"]}-{phase}' if region == 'clean' else f'{s["name"]}-{region}-{phase}'
                cells.append(dict(condition=name, image_id=image, phase=phase, region=region,
                    input_region=region if phase == 'image' else 'clean', position=s['position'],
                    budget=s['budget'], row_start=s['row_start'], owner=s['owner'], population='supplied_native_prefix',
                    mode='natural' if phase == 'clean' else 'intervention',
                    observations=[s['position']], patch_layers=[2, 13, 27] if phase == 'noop' else
                    [BOUNDARIES[phase]] if phase in BOUNDARIES else []))
    return cells


def tensor_identity(value):
    import torch
    t = value.detach().cpu().contiguous()
    return dict(shape=list(t.shape), dtype=str(t.dtype),
                sha256=hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest())


def batch_identity(batch):
    import torch
    return dict(prompt_token_ids=list(batch.prompt_token_ids[0]), image_grid_thw=list(batch.image_grids[0]),
        media_sha256=batch.media_sha256[0], tensors={k: tensor_identity(v) for k, v in batch.inputs.items()
            if isinstance(v, torch.Tensor)})


def rectangle_mask(shape, rect):
    import numpy as np
    h, w = shape[:2]
    x1, y1, x2, y2 = rect
    if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
        raise ValueError('rectangle outside original canvas')
    mask = np.zeros((h, w), dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask


def mask_statistics(rgb, site):
    import numpy as np
    means, counts, rings, l1 = [], [], [], []
    for region in ('target', 'background'):
        rect = site[region]
        x1, y1, x2, y2 = rect
        outer = [max(0, x1 - 8), max(0, y1 - 8), min(rgb.shape[1], x2 + 8), min(rgb.shape[0], y2 + 8)]
        mask = rectangle_mask(rgb.shape, rect)
        ring = rectangle_mask(rgb.shape, outer) & ~mask
        means.append(rgb[ring].mean(0).tolist()); counts.append(int(ring.sum())); rings.append(outer)
        l1.append(int(np.abs(rgb[mask].astype('int64') - np.array(site['fill'])).sum()))
    fill = np.floor(np.array(means).mean(0) + .5).astype('uint8').tolist()
    if fill != site['fill'] or counts != site['ring_counts'] or l1 != site['pixel_L1']:
        raise ValueError('frozen source-pixel fill/ring/L1 differs')
    return dict(fill=fill, ring_means=means, ring_counts=counts, ring_rectangles=rings, pixel_L1=l1)


def prepare_pixels(directory, image, site):
    import numpy as np
    from PIL import Image, ImageDraw
    from src.data.geometry import coord_bins_to_pixel_xyxy
    from src.qwen.images import rgb_image_sha256
    with Image.open(image['image_path']) as source:
        clean = np.array(source.convert('RGB'))
    owner = owner_entry.OWNERS[site['owner']]
    expected = list(coord_bins_to_pixel_xyxy(owner['box'], image_width=1248, image_height=832, field='target'))
    if clean.shape != (832, 1248, 3) or expected != site['target']:
        raise ValueError('source geometry/target norm1000 conversion differs')
    stats = mask_statistics(clean, site)
    variants, views, details = {}, [], []
    for region in ('clean', 'target', 'background'):
        rgb = clean.copy()
        mask = rectangle_mask(clean.shape, site[region]) if region != 'clean' else np.zeros(clean.shape[:2], dtype=bool)
        if region != 'clean':
            rgb[mask] = site['fill']
        stem = directory / f'{image["image_id"]}-{region}'
        png = Image.fromarray(rgb)
        png.save(stem.with_suffix('.png'))
        np.save(stem.with_suffix('.npy'), rgb, allow_pickle=False)
        np.save(Path(str(stem) + '-mask.npy'), mask, allow_pickle=False)
        Image.fromarray(mask.astype('uint8') * 255).save(Path(str(stem) + '-mask.png'))
        variants[region] = dict(image=binding(stem.with_suffix('.png')), array=binding(stem.with_suffix('.npy')),
            mask=binding(Path(str(stem) + '-mask.npy')), mask_image=binding(Path(str(stem) + '-mask.png')),
            pixel_sha256=rgb_image_sha256(png), changed_pixel_count=int(mask.sum()))
        view = png.copy()
        draw = ImageDraw.Draw(view)
        for label, color in [('target', 'red'), ('background', 'lime')]:
            x1, y1, x2, y2 = site[label]
            draw.rectangle([x1, y1, x2 - 1, y2 - 1], outline=color, width=3)
            draw.text((x1, max(0, y1 - 12)), label, fill=color)
        view.thumbnail((624, 416)); views.append(view)
        detail_views = []
        for label, color in [('target', 'red'), ('background', 'lime')]:
            x1, y1, x2, y2 = site[label]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            left, top = min(max(0, cx - 64), 1120), min(max(0, cy - 64), 704)
            crop = png.crop((left, top, left + 128, top + 128))
            ImageDraw.Draw(crop).rectangle([x1 - left, y1 - top, x2 - left - 1, y2 - top - 1], outline=color)
            detail_views.append(crop.resize((256, 256), Image.Resampling.NEAREST))
        details.append(detail_views)
    gallery = Image.new('RGB', (1872, 1046), 'white')
    for i, (name, view) in enumerate(zip(('clean', 'target', 'background'), views)):
        gallery.paste(view, (624 * i, 30)); ImageDraw.Draw(gallery).text((624 * i + 10, 8), f'{image["image_id"]} {name}', fill='black')
        for j, label in enumerate(('target', 'background')):
            gallery.paste(details[i][j], (624 * i, 474 + j * 286))
            ImageDraw.Draw(gallery).text((624 * i + 265, 484 + j * 286), label + ' 2x nearest', fill='black')
    gallery.save(directory / f'{image["image_id"]}-gallery.png')
    return dict(variants=variants, statistics=stats, gallery=binding(directory / f'{image["image_id"]}-gallery.png'))


def verify_pixels(packet):
    import numpy as np
    from PIL import Image
    from src.qwen.images import rgb_image_sha256
    for image, site in SITES.items():
        record = packet['pixels'][str(image)]
        clean = np.load(record['variants']['clean']['array']['path'], allow_pickle=False)
        if clean.shape != (832, 1248, 3) or clean.dtype != np.uint8 or mask_statistics(clean, site) != record['statistics']:
            raise ValueError('prepared RGB statistics differ')
        if record['variants']['clean']['pixel_sha256'] != packet['original_media'][str(image)]:
            raise ValueError('lossless clean RGB differs from qualified original media')
        for region, v in record['variants'].items():
            for k in ('image', 'array', 'mask', 'mask_image'):
                if a.digest(v[k]['path']) != v[k]['sha256']:
                    raise ValueError('prepared image/mask bytes changed')
            rgb = np.load(v['array']['path'], allow_pickle=False)
            mask = np.load(v['mask']['path'], allow_pickle=False)
            expected = rectangle_mask(clean.shape, site[region]) if region != 'clean' else np.zeros(clean.shape[:2], dtype=bool)
            with Image.open(v['image']['path']) as im:
                decoded = np.array(im.convert('RGB')); pixel_sha = rgb_image_sha256(im)
            with Image.open(v['mask_image']['path']) as im:
                mask_png = np.array(im)
            if (rgb.dtype != np.uint8 or rgb.shape != clean.shape or mask.dtype != bool or
                    not np.array_equal(mask, expected) or not np.array_equal(mask_png, expected.astype('uint8') * 255) or
                    not np.array_equal(decoded, rgb) or not np.array_equal(rgb[~mask], clean[~mask]) or
                    pixel_sha != v['pixel_sha256'] or v['changed_pixel_count'] != int(expected.sum()) or
                    region != 'clean' and not np.all(rgb[mask] == site['fill'])):
                raise ValueError('half-open mask/complement/lossless RGB differs')


def validate_conditions(packet):
    if (packet.get('schema') != SCHEMA or packet.get('bounds') != BOUNDS or packet.get('sites') != {str(k): v for k, v in SITES.items()} or
            packet.get('coordinate_ids') != COORD_IDS or packet.get('model_loaded') is not False or len(packet['conditions']) != 20):
        raise ValueError('frozen matrix/resources differs')
    for key, variants in packet['requests'].items():
        if key not in packet['sites'] or set(variants) != {'clean', 'target', 'background'}:
            raise ValueError('frozen image-condition inputs differ')
        for region, request in variants.items():
            image = packet['pixels'][key]['variants'][region]
            if (request['image_id'] != int(key) or request['image_path'] != image['image']['path'] or
                    request['image_sha256'] != image['image']['sha256'] or request['media_sha256'] != image['pixel_sha256'] or
                    request['crop'] != [0, 0, 1248, 832] or request['view_scale'] != 1 or
                    request['width'] != 1248 or request['height'] != 832 or
                    request['prompt_token_ids'] != variants['clean']['prompt_token_ids'] or
                    request['image_grid_thw'] != variants['clean']['image_grid_thw'] or
                    packet['processor'][key][region]['prompt_token_ids'] != request['prompt_token_ids'] or
                    packet['processor'][key][region]['image_grid_thw'] != request['image_grid_thw'] or
                    packet['processor'][key][region]['media_sha256'] != request['media_sha256']):
                raise ValueError('frozen text/grid/geometry/media binding differs')
    for cell, declared in zip(packet['conditions'], definitions(), strict=True):
        if any(cell.get(k) != v for k, v in declared.items()):
            raise ValueError('frozen action/boundary/donor selector differs')
        if (len(cell['expected_ids']) != cell['budget'] or cell['expected_ids'][cell['position'] - 1] != 151648 or
                cell['forced_actions'] != {str(i): v for i, v in enumerate(cell['expected_ids'][:cell['position']])} or
                set(cell['selectors']) != {str(cell['position'])} or
                len(packet['requests'][str(cell['image_id'])]['clean']['prompt_token_ids']) != 1362):
            raise ValueError('literal pre-x1 prefix/context differs')


def prepare(directory):
    from probes import rollout_row_credit as retained, iterative_positive as p
    from probes.rule_stability.__main__ import runtime_identity
    oldpath = owner_entry.OUTPUT / 'prepared-01/input-packet.json'
    old = a.load(oldpath)
    if old['runtime'] != runtime_identity():
        raise ValueError('qualified current runtime differs')
    for key, b in old['bindings'].items():
        if a.digest(b['path']) != b['sha256']:
            raise ValueError('qualified immutable input changed:' + key)
    continuity.check_payloads(old['payloads'])
    pipeline = dict(old['cached_pipeline_files'])
    for path, sha in pipeline.items():
        if a.digest(ROOT / path) != sha:
            raise ValueError('qualified cached source changed:' + path)
    for path in ('src/qwen/inspection.py', 'src/qwen/native.py', 'src/qwen/images.py', 'probes/coordinate_diagnostics/owner_entry.py'):
        pipeline[path] = a.digest(ROOT / path)
    q = retained.frontend()
    if q.model is not None:
        raise ValueError('CPU preparation loaded a model')
    directory = Path(directory).resolve()
    if not directory.is_relative_to(OUTPUT):
        raise ValueError('preparation outside task owner')
    directory.mkdir(parents=True, exist_ok=False)
    pixels, requests, processor = {}, {}, {}
    for image, site in SITES.items():
        key = str(image)
        original = readout.verify_frontend(q, old['images'][key], old['requests'][key])
        pixels[key] = prepare_pixels(directory, old['images'][key], site)
        requests[key], processor[key] = {}, {}
        for region, v in pixels[key]['variants'].items():
            request = dict(old['requests'][key], image_path=v['image']['path'], image_sha256=v['image']['sha256'],
                request_id=f'visual-state:{image}:{region}', media_sha256=v['pixel_sha256'])
            request['original_encoding_sha256'] = request.pop('encoding_sha256')
            batch = p.native_request(request, p.load(p.POLICY), q.processor)
            identity = batch_identity(batch)
            if (identity['prompt_token_ids'] != list(original.prompt_token_ids[0]) or
                    identity['image_grid_thw'] != list(original.image_grids[0]) or identity['media_sha256'] != v['pixel_sha256']):
                raise ValueError('perturbed prompt/grid/media differs')
            if region == 'clean' and identity['tensors'] != batch_identity(original)['tensors']:
                raise ValueError('lossless clean processor differs')
            requests[key][region], processor[key][region] = request, identity
    cells = definitions()
    for cell in cells:
        reference = next(c for c in old['conditions'] if c['mode'] == 'natural' and c['image_id'] == cell['image_id'])
        for k in ('expected_ids', 'saved_raw_logprobs', 'saved_policy_logprobs', 'target_box'):
            cell[k] = reference[k]
        cell['forced_actions'] = {str(i): v for i, v in enumerate(cell['expected_ids'][:cell['position']])}
        cell['selectors'] = {str(cell['position']): reference['selectors'][str(cell['position'])]}
    bindings = dict(old['bindings'], protocol=binding(UNIT / 'unit.md'), prior_owner_packet=binding(oldpath))
    packet = dict(schema=SCHEMA, conditions=cells, sites={str(k): v for k, v in SITES.items()}, bounds=BOUNDS,
        images=old['images'], requests=requests, pixels=pixels, processor=processor, bindings=bindings,
        original_media={k: v['media_sha256'] for k, v in old['requests'].items()},
        runtime=old['runtime'], payloads=old['payloads'], cached_pipeline_files=pipeline,
        model_loaded=False, coordinate_ids=COORD_IDS, eos_id=old['eos_id'], object_start_id=old['object_start_id'], producer=producer_identity())
    validate_conditions(packet); verify_pixels(packet)
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
    continuity.check_payloads(packet['payloads']); verify_pixels(packet)
    return config, packet


class CellInvalid(ValueError):
    """A failed technical cell whose dependent interpretation must be held."""


def donor_name(cell):
    stem = SITES[cell['image_id']]['name']
    return f'{stem}-clean' if cell['phase'] == 'noop' else f'{stem}-{cell["region"]}-image'


def validate_donor(cell, donor, width):
    if not cell['patch_layers']:
        if donor is not None:
            raise CellInvalid('unrequested donor')
        return
    if donor is None:
        raise CellInvalid('missing saved donor')
    record, tensors = donor
    meta = record['capture']
    if (record['condition'] != donor_name(cell) or record['image_id'] != cell['image_id'] or
            meta['position'] != cell['position'] or meta['prompt_width'] != width or
            meta['prefix_token_ids'] != cell['expected_ids'][:cell['position']] or
            meta['token_id'] != 151648 or meta['boundaries'] != [2, 13, 27]):
        raise CellInvalid('saved donor image/action/prefix/boundary differs')
    for b in (record['artifact'], record['capture']['tensors']):
        if a.digest(b['path']) != b['sha256']:
            raise CellInvalid('saved donor bytes changed')
    if a.load(record['artifact']['path']) != {k: v for k, v in record.items() if k != 'artifact'}:
        raise CellInvalid('donor fields differ from saved bound artifact')
    for layer in cell['patch_layers']:
        key = f'residual_{layer}'
        if tensor_identity(tensors[key]) != meta['tensor_identities'][key]:
            raise CellInvalid('saved donor tensor/layer identity differs')


def replace_current(hidden, donor):
    """Clone only for replacement; preserve every unselected tensor row."""
    if hidden.ndim != 3 or hidden.shape[0] != 1 or donor.shape != hidden[0, -1].shape or donor.dtype != hidden.dtype:
        raise CellInvalid('donor dtype/shape differs from receiver')
    result = hidden.clone()
    result[0, -1] = donor.to(hidden.device)
    return result


class ResidualCapture:
    """Pair actual embedding/cache/boundary/head calls with the raw processor."""
    def __init__(self, model, cell, width, donor=None, *, hidden_size=2048):
        from src.qwen.inspection import resolve_text_stack
        self.stack = resolve_text_stack(model)
        if len(self.stack.layers) != 28:
            raise CellInvalid('expected exactly28 decoder layers')
        self.cell, self.width, self.hidden_size = cell, width, hidden_size
        validate_donor(cell, donor, width)
        self.donor = donor
        self.calls = self.bound = self.embed_calls = 0
        self.active, self.embedded, self.evidence = False, None, None
        self.tensors, self.consumed, self.handles = {}, [], []
        self.model = model

    def embed(self, module, args, output):
        if not args or args[0].ndim != 2 or args[0].shape[0] != 1 or self.embedded is not None:
            raise CellInvalid('embedding/current action call alignment differs')
        self.embedded = args[0].detach().cpu().clone()
        self.embed_calls += 1
        return None

    def before(self, module, args, kwargs):
        if self.active or self.calls != self.bound or self.embedded is None:
            raise CellInvalid('shifted language/action call')
        self.active = True
        if self.calls == self.cell['position']:
            cache = kwargs.get('past_key_values')
            before = cache.get_seq_length() if cache is not None else None
            expected = self.width + self.cell['position'] - 1
            if (self.embedded.shape != (1, 1) or self.embedded.item() != 151648 or before != expected or
                    kwargs.get('deepstack_visual_embeds') is not None or
                    kwargs.get('visual_pos_masks') is not None and bool(kwargs['visual_pos_masks'].any())):
                raise CellInvalid('actual pre-x1 token/cache/visual injection differs')
            self.cache = cache
            self.evidence = dict(position=self.calls, token_id=int(self.embedded.item()),
                prompt_width=self.width, cache_before=before, current_token_length=1,
                deepstack_absent=True, boundaries=[2, 13, 27], consumed=[],
                position_ids=kwargs['position_ids'].detach().cpu().tolist() if kwargs.get('position_ids') is not None else None)
        return None

    def after(self, module, args, kwargs, output):
        if not self.active:
            raise CellInvalid('language output without paired input')
        if self.calls == self.cell['position']:
            cache = output.past_key_values
            if cache is not self.cache or cache.get_seq_length() != self.width + self.cell['position']:
                raise CellInvalid('receiver cache identity/length changed')
            self.evidence['cache_after'] = cache.get_seq_length()
            self.evidence['receiver_cache_retained'] = True
        self.active = False; self.calls += 1; self.embedded = None
        return None

    def boundary(self, layer, module, args, kwargs):
        import torch
        if not self.active or self.calls != self.cell['position']:
            return None
        hidden = args[0] if args else kwargs.get('hidden_states')
        if hidden is None or hidden.shape != (1, 1, self.hidden_size) or hidden.dtype != torch.bfloat16:
            raise CellInvalid('native residual shape/dtype differs')
        self.consumed.append(layer)
        effective = hidden
        if layer in self.cell['patch_layers']:
            donor = self.donor[1][f'residual_{layer}']
            effective = replace_current(hidden, donor)
            self.evidence['consumed'].append(dict(layer=layer, donor_condition=self.donor[0]['condition'],
                donor_artifact=self.donor[0]['artifact'], receiver_cache_retained=True,
                changed_rows=[[0, 0]], other_arguments_unchanged=True, original_tensor_unmodified=True))
        self.tensors[f'residual_{layer}'] = effective[0, -1].detach().cpu().clone()
        if effective is hidden:
            return None
        if args:
            return (effective, *args[1:]), kwargs
        return args, dict(kwargs, hidden_states=effective)

    def head(self, module, args, output):
        if self.calls - 1 != self.cell['position']:
            return None
        if not args or args[0].shape != (1, 1, self.hidden_size) or output.ndim != 3 or output.shape[:2] != (1, 1):
            raise CellInvalid('selected normalized head/action differs')
        self.tensors['normalized_h'] = args[0][0, -1].detach().cpu().clone()
        self.tensors['raw'] = output[0, -1].detach().float().cpu().clone()
        return None

    def bind(self, input_ids, scores, *, median=False):
        import torch
        pos = input_ids.shape[1] - self.width
        if pos != self.calls - 1 or (not median and pos != self.bound):
            raise CellInvalid('shifted forward/score action binding')
        if pos == self.cell['position']:
            prefix = input_ids[0, self.width:].tolist()
            if prefix != self.cell['expected_ids'][:pos] or self.consumed != [2, 13, 27]:
                raise CellInvalid('selected literal prefix/boundary consumption differs')
            if not median and not torch.equal(self.tensors['raw'], scores[0].detach().float().cpu()):
                raise CellInvalid('observed native head differs from raw score processor')
            self.evidence['prefix_token_ids'] = prefix
            if median:
                self.tensors['median'] = scores[0].detach().float().cpu().clone()
        if not median:
            self.bound += 1

    def finish(self, tokens):
        if self.calls != len(tokens) or self.bound != len(tokens) or self.embed_calls != len(tokens) or self.active:
            raise CellInvalid('unconsumed forward/embedding/action')
        if self.evidence is None or set(self.tensors) != {'residual_2', 'residual_13', 'residual_27', 'normalized_h', 'raw', 'median'}:
            raise CellInvalid('missing selected pre-x1 state')
        self.evidence.update(hook_calls=self.calls, tensor_identities={k: tensor_identity(v) for k, v in self.tensors.items()},
            native_head_raw_exact=True, hook_removed=True, full_vocabulary_size=self.tensors['raw'].numel())
        return self.evidence

    def __enter__(self):
        from functools import partial
        try:
            self.handles.append(self.model.get_input_embeddings().register_forward_hook(self.embed))
            self.handles.append(self.stack.language_model.register_forward_pre_hook(self.before, with_kwargs=True))
            self.handles.append(self.stack.language_model.register_forward_hook(self.after, with_kwargs=True))
            for layer, module in ((2, self.stack.layers[3]), (13, self.stack.layers[14]), (27, self.stack.norm)):
                self.handles.append(module.register_forward_pre_hook(partial(self.boundary, layer), with_kwargs=True))
            self.handles.append(self.stack.head.register_forward_hook(self.head))
            return self
        except Exception:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *args):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class Observer(continuity.ContinuationProcessor):
    def __init__(self, *args, capture, **kwargs):
        super().__init__(*args, **kwargs)
        self.capture = capture

    def __call__(self, input_ids, scores):
        self.last_history = input_ids[0, self.width:].tolist()
        self.capture.bind(input_ids, scores, median=self.raw is not None)
        return super().__call__(input_ids, scores)


class NativeSession(readout.NativeSession):
    def __init__(self, checkpoint, packet):
        # The maintained loader expects one original request per image.
        original = dict(packet, requests={k: v['clean'] for k, v in packet['requests'].items()})
        super().__init__(checkpoint, original)
        from probes import iterative_positive as p
        self.packet = packet
        self.batches = {}
        for image, variants in packet['requests'].items():
            for region, request in variants.items():
                batch = p.native_request(request, p.load(p.POLICY), self.q.processor)
                if batch_identity(batch) != packet['processor'][image][region]:
                    raise ValueError('executed processor tensor/media identity differs')
                self.batches[(image, region)] = batch

    def generate(self, cell, donor=None):
        import torch
        from src.qwen.generation import generate_continuations, NativeGenerationPolicy
        batch = self.batches[(str(cell['image_id']), cell['input_region'])]
        width = len(batch.prompt_token_ids[0])
        capture = ResidualCapture(self.q.model, cell, width, donor, hidden_size=getattr(self, 'fixture_hidden_size', 2048))
        raw = Observer(cell, self.packet, width, capture=capture)
        median = Observer(cell, self.packet, width, raw=raw, capture=capture)
        begin, self.failure_evidence = time.monotonic(), None
        try:
            with capture, torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                result = generate_continuations(self.q.model, batch, extensions=[()], budgets=[cell['budget']],
                    eos_token_id=self.packet['eos_id'], pad_token_id=self.q.tokenizer.pad_token_id,
                    policy=NativeGenerationPolicy(temperature=0, top_p=1, top_k=0, repetition_penalty=1, use_model_defaults=False),
                    trace='raw_and_policy', allow_pad_tokens=True,
                    logits_processor=[raw, self.norm.generation_transform(), median])[0]
                meta = capture.finish(list(result.token_ids))
        except Exception:
            self.failure_evidence = dict(condition=cell['condition'], phase='cached_generation_or_state_binding',
                observed_prefix=getattr(raw, 'last_history', []), selected_unconfirmed=raw.emitted,
                raw_steps=raw.steps, median_steps=median.steps, capture=capture.evidence)
            raise
        image = self.packet['images'][str(cell['image_id'])]
        return dict(request_id=batch.request_ids[0], width=image['width'], height=image['height'],
            token_ids=list(result.token_ids), text=self.q.tokenizer.decode(result.token_ids, skip_special_tokens=False),
            stop_reason=result.stop_reason, raw_logprobs=list(result.raw_logprobs), policy_logprobs=list(result.policy_logprobs),
            raw_steps=raw.steps, median_steps=median.steps, raw_observations=raw.observations,
            median_observations=median.observations, generation_seconds=time.monotonic() - begin,
            capture=meta, _tensors=capture.tensors)


def load_capture(record):
    from safetensors.torch import load_file
    b = record['capture']['tensors']
    if a.digest(b['path']) != b['sha256']:
        raise CellInvalid('saved selected state bytes changed')
    tensors = load_file(b['path'], device='cpu')
    if {k: tensor_identity(v) for k, v in tensors.items()} != record['capture']['tensor_identities']:
        raise CellInvalid('saved selected state tensor identities differ')
    return tensors


def validate_record(record, cell, packet):
    import torch
    continuity.validate_record(record, cell, packet)
    meta, tensors = record['capture'], load_capture(record)
    expected = dict(position=cell['position'], token_id=151648, prompt_width=1362, current_token_length=1,
        cache_before=1362 + cell['position'] - 1, cache_after=1362 + cell['position'], deepstack_absent=True,
        boundaries=[2, 13, 27], receiver_cache_retained=True, native_head_raw_exact=True, hook_removed=True)
    if any(meta.get(k) != v for k, v in expected.items()) or meta['prefix_token_ids'] != record['token_ids'][:cell['position']]:
        raise CellInvalid('saved actual cache/action/prefix differs')
    if [c['layer'] for c in meta['consumed']] != cell['patch_layers'] or meta['hook_calls'] != len(record['token_ids']):
        raise CellInvalid('saved patch consumption differs')
    for c in meta['consumed']:
        if (c['donor_condition'] != donor_name(cell) or c['changed_rows'] != [[0, 0]] or
                any(c.get(k) is not True for k in ('receiver_cache_retained', 'other_arguments_unchanged', 'original_tensor_unmodified'))):
            raise CellInvalid('saved donor/receiver mutation differs')
        if a.digest(c['donor_artifact']['path']) != c['donor_artifact']['sha256']:
            raise CellInvalid('saved donor artifact changed')
    for channel in ('raw', 'median'):
        obs = record[f'{channel}_observations'][0]
        if tensors[channel][COORD_START:COORD_START + 1000].tolist() != obs['coordinate_scores'] or not bool(torch.isfinite(tensors[channel]).all()):
            raise CellInvalid('saved full/coordinate score support differs')
        compact = readout.capture_scores(tensors[channel][None], packet)
        if any(obs.get(k) != v for k, v in compact.items()):
            raise CellInvalid('saved full-vocabulary normalizer/winner differs')


def equality(left, right, keys):
    import torch
    comparisons = {k: dict(exact=torch.equal(left[k], right[k]),
        max_absolute_difference=float((left[k].double() - right[k].double()).abs().max())) for k in keys}
    return dict(qualified=all(c['exact'] for c in comparisons.values()), comparisons=comparisons, tolerance=None)


def control(record, cell, records):
    stem = SITES[cell['image_id']]['name']
    if cell['phase'] == 'clean':
        return continuity.fidelity(record, cell)
    if cell['phase'] == 'noop':
        clean = records[f'{stem}-clean']
        report = equality(load_capture(record), load_capture(clean), ('raw', 'median', 'normalized_h'))
        report['sequence_channels_exact'] = all(record[k] == clean[k] for k in
            ('token_ids', 'raw_logprobs', 'policy_logprobs', 'raw_steps', 'median_steps'))
        report['qualified'] &= report['sequence_channels_exact']
        return report
    if cell['phase'] == 'final':
        return equality(load_capture(record), load_capture(records[donor_name(cell)]), ('normalized_h', 'raw', 'median'))
    return dict(qualified=True, meaning='acquisition/transport; no scientific effect-size gate')


def dependency(cell, qualified):
    stem = SITES[cell['image_id']]['name']
    needs = []
    if cell['phase'] != 'clean':
        needs.append(f'{stem}-clean')
    if cell['phase'] not in ('clean', 'noop'):
        needs.append(f'{stem}-noop')
    if cell['phase'] in BOUNDARIES:
        needs.append(donor_name(cell))
    if cell['phase'] in ('early', 'middle'):
        needs.append(f'{stem}-{cell["region"]}-final')
    return [name for name in needs if not qualified.get(name, False)]


def scores(record, cell):
    result, tensors = {}, load_capture(record)
    for channel in ('raw', 'median'):
        o, step = record[f'{channel}_observations'][0], record[f'{channel}_steps'][cell['position']]
        values = o['coordinate_scores']; target = cell['target_box'][0]
        result[channel] = dict(continuity.score_summary(o, step),
            primary_target_minus_endpoint0=values[target] - values[0],
            target_minus_best_other_full=values[target] - max(
                max(v for i, v in enumerate(values) if i != target), o['top5_noncoordinate'][0]['score']),
            target_full_rank=1 + int((tensors[channel] > values[target]).sum()),
            target_coordinate_rank=1 + sum(v > values[target] for v in values),
            coordinate_scores=values, full_log_normalizer=o['full_log_normalizer'],
            additional_fixed_contrasts=({'s495_minus_s0': values[495] - values[0]} if cell['image_id'] == 351017 else
                {'s544_minus_s0': values[544] - values[0], 's546_minus_s544': values[546] - values[544]}))
    return result


def state_distances(record, clean):
    import torch
    left, right = load_capture(record), load_capture(clean)
    return {str(layer): dict(L2=float(torch.linalg.vector_norm(left[f'residual_{layer}'].double() - right[f'residual_{layer}'].double())),
        cosine=float(torch.nn.functional.cosine_similarity(left[f'residual_{layer}'].double(), right[f'residual_{layer}'].double(), dim=0)),
        descriptive_only=True, intermediate_head_scores=False) for layer in (2, 13, 27)}


def contrasts(rows):
    byname = {r['condition']: r for r in rows if r['status'] == 'completed'}
    result = []
    for image, site in SITES.items():
        stem = site['name']
        for phase in ('image', 'final', 'early', 'middle'):
            names = [f'{stem}-clean', f'{stem}-target-{phase}', f'{stem}-background-{phase}']
            if not all(n in byname for n in names):
                result.append(dict(image_id=image, phase=phase, status='HOLD', unavailable_cells=[n for n in names if n not in byname]))
                continue
            for channel in ('raw', 'median'):
                clean, target, background = [byname[n]['scores'][channel]['primary_target_minus_endpoint0'] for n in names]
                result.append(dict(image_id=image, phase=phase, channel=channel,
                    estimand='V' if phase == 'image' else 'P', construction_control=phase == 'final',
                    boundary=BOUNDARIES.get(phase), clean=clean, target_minus_clean=target - clean,
                    background_minus_clean=background - clean, target_minus_background=target - background,
                    abs_target_minus_clean=abs(target - clean), abs_background_minus_clean=abs(background - clean),
                    abs_target_minus_background=abs(target - background), effect_size_gate=False))
    return result


def consume(output, packet, tokenizer):
    validate_conditions(packet); verify_pixels(packet)
    output = Path(output)
    index = a.load(output / 'conditions.json')
    if [e['condition'] for e in index] != [c['condition'] for c in packet['conditions']]:
        raise ValueError('saved consumer condition order differs')
    records, rows, qualified = {}, [], {}
    for entry, cell in zip(index, packet['conditions'], strict=True):
        name = cell['condition']; unmet = dependency(cell, qualified)
        if entry['status'] == 'skipped-HOLD':
            if not unmet and not entry['reason'].startswith(('resource_limit', 'execution_failure')):
                raise ValueError('unjustified cell HOLD')
            rows.append(dict(condition=name, status=entry['status'], reason=entry['reason']))
            qualified[name] = False
            continue
        if entry['status'] == 'attempted-invalid':
            if 'filename' not in entry or not (output / entry['filename']).is_file():
                raise ValueError('missing failed-cell evidence')
            qualified[name] = False
            rows.append(dict(condition=name, status=entry['status'], reason=entry['reason'], artifact=binding(output / entry['filename'])))
            continue
        if entry['status'] != 'completed' or unmet or entry['filename'] != name + '.json':
            raise ValueError('artifact published after dependency HOLD/identity drift')
        path = output / entry['filename']; record = a.load(path)
        if (record.get('schema') != SCHEMA or record.get('condition') != name or record.get('image_id') != cell['image_id'] or
                record['request_id'] != packet['requests'][str(cell['image_id'])][cell['input_region']]['request_id']):
            raise ValueError('condition/input/request identity differs')
        validate_record(record, cell, packet)
        record['artifact'] = binding(path)
        if cell['patch_layers']:
            donor = records[donor_name(cell)]
            validate_donor(cell, (donor, load_capture(donor)), 1362)
            for item in record['capture']['consumed']:
                if item['donor_artifact'] != donor['artifact']:
                    raise ValueError('consumed donor ownership differs')
        check = control(record, cell, records); qualified[name] = check['qualified']
        analysis = owner_entry.analyze(record, dict(cell, observations=list(range(cell['position'], cell['position'] + 4))), packet, tokenizer)
        records[name] = record
        rows.append(dict(condition=name, status='completed', phase=cell['phase'], region=cell['region'], image_id=cell['image_id'],
            artifact=binding(path), capture=record['capture'], control=check, scores=scores(record, cell),
            residual_distances_to_clean=state_distances(record, records[SITES[cell['image_id']]['name'] + '-clean']),
            analysis=analysis, generated_actions=len(record['token_ids'])))
    expected_files = {e[k] for e in index for k in ('filename', 'failure') if k in e}
    if {p.name for p in output.glob('bottle-*.json')} | {p.name for p in output.glob('person-*.json')} != expected_files:
        raise ValueError('extra/unconsumed condition artifact')
    return dict(schema=SCHEMA, complete=len(records) == 20 and all(qualified.values()), conditions=rows,
        completed_requests=len(records), attempted_requests=sum(e['status'] != 'skipped-HOLD' for e in index),
        generated_actions=sum(len(r['token_ids']) for r in records.values()),
        requested_cells=20, selected_states=2, requested_groups=dict(clean=2, noop=2, image=4, final=4, early=4, middle=4),
        contrasts=contrasts(rows), control_qualified=qualified,
        limitations='Residual transport with clean receiver KV retained; final transfer is construction control. No unique fault layer, physical recovery, prevalence, training or repair claim.')


def resource_excess(output, begin, *, native):
    values = readout.usage(output, native=native)
    return values, [k for k, v in values.items() if v > BOUNDS[k]] + (
        ['wall_seconds'] if time.monotonic() - begin > BOUNDS['wall_seconds'] else [])


def run(config_path, output, *, cpu_factory=None):
    from safetensors.torch import save_file
    output = Path(output).resolve()
    if not output.is_relative_to(OUTPUT):
        raise ValueError('invocation outside task owner')
    output.mkdir(parents=True, exist_ok=False)
    begin, session, packet = time.monotonic(), None, None
    native, status, code = cpu_factory is None, 'technical_HOLD', 2
    statuses, records, qualified = [], {}, {}
    counts = dict(checkpoint_loads=0, fixture_sessions=0, attempted_requests=0, completed_requests=0,
        generated_actions=0, optimizer=0, backward=0, replay=0, training=0, warmup=0, exports=0)
    a.write(output / 'invocation.json', dict(schema=SCHEMA, config=binding(config_path), pid=os.getpid(),
        started=time.time(), compute='native' if native else 'CPU_FIXTURE', retry='no_automatic_relaunch'))
    try:
        config, packet = load_packet(config_path, cpu=not native)
        if native and str(output) != config['output']:
            raise ValueError('output differs from exact release')
        a.write(output / 'qualification.json', dict(source_revision=config['source_revision'], producer=config['producer_files'],
            runtime=config['runtime'], inputs=packet['bindings'], processor=packet['processor'], payloads=packet['payloads']))
        _, exceeded = resource_excess(output, begin, native=native)
        if exceeded:
            raise ValueError('resource excess before load:' + ','.join(exceeded))
        start = time.monotonic(); counts['checkpoint_loads' if native else 'fixture_sessions'] = 1
        session = (cpu_factory or NativeSession)(PREVIOUS / 'checkpoint-16', packet)
        a.write(output / 'load-B16.json', dict(seconds=time.monotonic() - start, composition=session.composition))
        for cell in packet['conditions']:
            name = cell['condition']; unmet = dependency(cell, qualified)
            if unmet:
                statuses.append(dict(condition=name, status='skipped-HOLD', reason='dependency:' + ','.join(unmet)))
                continue
            _, exceeded = resource_excess(output, begin, native=native)
            if exceeded:
                break
            counts['attempted_requests'] += 1
            try:
                donor = records.get(donor_name(cell)) if cell['patch_layers'] else None
                record = session.generate(cell, (donor, load_capture(donor)) if donor else None)
                record.update(schema=SCHEMA, condition=name, image_id=cell['image_id'])
                tensors = record.pop('_tensors')
                tensor_path = output / f'{name}.safetensors'; save_file(tensors, str(tensor_path))
                record['capture']['tensors'] = binding(tensor_path)
                filename = name + '.json'; a.write(output / filename, record)
                counts['generated_actions'] += len(record['token_ids'])
                validate_record(record, cell, packet)
                check = control(record, cell, records)
                qualified[name] = check['qualified']
                record['artifact'] = binding(output / filename); records[name] = record
                counts['completed_requests'] += 1
                statuses.append(dict(condition=name, status='completed', filename=filename))
                a.write(output / f'control-{name}.json', check)
            except CellInvalid as error:
                qualified[name] = False
                failure = name + '-invalid.json'
                a.write(output / failure, dict(error=str(error), partial=getattr(session, 'failure_evidence', None),
                    phase='instrumentation_or_saved_binding', attempted_requests=counts['attempted_requests']))
                published = output / (name + '.json')
                statuses.append(dict(condition=name, status='attempted-invalid', reason=str(error),
                    filename=published.name if published.exists() else failure, failure=failure))
            a.write(output / f'resource-{counts["attempted_requests"]:02d}.json',
                dict(seconds=time.monotonic() - begin, counts=counts, usage=resource_excess(output, begin, native=native)[0]))
        done = {e['condition'] for e in statuses}
        statuses.extend(dict(condition=c['condition'], status='skipped-HOLD', reason='resource_limit:' + ','.join(exceeded))
            for c in packet['conditions'] if c['condition'] not in done)
        a.write(output / 'conditions.json', statuses)
        tokenizer = session.q.tokenizer; session.close(); session = None
        report = consume(output, packet, tokenizer); report['compute'] = 'native' if native else 'CPU_FIXTURE'
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
            statuses.extend(dict(condition=c['condition'], status='skipped-HOLD', reason='execution_failure:' + type(error).__name__)
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
        proposal = prepare(args.output)
        print(json.dumps(dict(status='CPU_PREPARED', input_packet=proposal['input_packet'], model_loaded=False)))
        return 0
    if args.command == 'run':
        if args.config is None:
            parser.error('run needs --config')
        return run(args.config, args.output, cpu_factory=cpu_factory)
    invocation = a.load(args.output / 'invocation.json')
    if a.digest(invocation['config']['path']) != invocation['config']['sha256']:
        raise ValueError('saved release changed')
    config = a.load(invocation['config']['path'])
    if a.digest(config['input_packet']['path']) != config['input_packet']['sha256']:
        raise ValueError('saved packet changed')
    from probes import rollout_row_credit as retained
    report = consume(args.output, a.load(config['input_packet']['path']), retained.frontend().tokenizer)
    if report != {k: v for k, v in a.load(args.output / 'readback.json').items() if k != 'compute'}:
        raise ValueError('fresh saved consumer differs')
    print(json.dumps(dict(complete=report['complete'], completed_requests=report['completed_requests'],
        generated_actions=report['generated_actions'], control_qualified=report['control_qualified'])))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
