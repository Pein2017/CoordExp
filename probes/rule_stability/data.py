"""Frozen full-label inputs and row-then-image positive supervision."""
from __future__ import annotations

from pathlib import Path

from probes import iterative_positive as p
from probes import rollout_row_credit as retained
from probes.full_label_fit import experiment as full
from probes.full_label_fit import rollout

FULL_LABEL_PATH = full.INPUT_DIR / 'full-labels.json'
INPUT_MANIFEST_PATH = full.INPUT_DIR / 'manifest.json'
FULL_LABEL_SHA256 = '1cfdeb3bba14bb26034dd5dc4245a5c10ab240e1e780acdfc1edbdf6c32d4792'
INPUT_MANIFEST_SHA256 = '51ad01b2cf599087e4d93b269a0abb9d5c6be83625e9fe9c5f8541731bd962ad'
IMAGE_COUNT = 18
LABEL_COUNT = 570


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _digest(value):
    return (isinstance(value, str) and len(value) == 64
            and all(c in '0123456789abcdef' for c in value))


def validate_inputs(images, manifest):
    """Validate all570 labels and actual media/prompt bindings; select no owner subset."""
    _require(isinstance(images, list) and len(images) == IMAGE_COUNT, 'full-label input must contain all18 images')
    _require(isinstance(manifest, dict) and manifest.get('schema') == 'full-label-self-rollout-inputs-v1',
             'original full-label manifest schema changed')
    spec = manifest.get('full_labels', {})
    _require(spec.get('sha256') == FULL_LABEL_SHA256 and spec.get('images') == IMAGE_COUNT
             and spec.get('objects') == LABEL_COUNT and spec.get('path') == 'inputs/full-labels.json',
             'full-label manifest binding changed')
    bindings = manifest.get('images')
    _require(isinstance(bindings, list) and len(bindings) == IMAGE_COUNT, 'full-label image bindings changed')
    categories, ids, annotation_keys = {}, [], set()
    for image, binding in zip(images, bindings, strict=True):
        _require(isinstance(image, dict) and set(image) == set(full.FULL_FIELDS), 'full-label image schema changed')
        image_id = image['image_id']
        _require(type(image_id) is int and image_id not in ids, 'invalid or duplicate full-label image ID')
        ids.append(image_id)
        _require(isinstance(binding, dict) and type(binding.get('image_id')) is int
                 and binding['image_id'] == image_id, 'original manifest image order/identity changed')
        _require(image['cohort'] in ('human13', 'refined5') and binding.get('cohort') == image['cohort'],
                 'full-label image cohort changed')
        _require(type(image['width']) is int and type(image['height']) is int
                 and min(image['width'], image['height']) > 0, 'invalid full-label image dimensions')
        _require(isinstance(image['image_path'], str) and Path(image['image_path']).is_absolute(),
                 'full-label image path must be absolute')
        _require(_digest(image['image_sha256']) and binding.get('image_sha256') == image['image_sha256'],
                 'full-label image digest binding changed')
        prompt, grid = binding.get('prompt_token_ids'), binding.get('image_grid_thw')
        _require(isinstance(prompt, list) and prompt and all(type(t) is int and t >= 0 for t in prompt),
                 'invalid original prompt token IDs')
        _require(isinstance(grid, list) and len(grid) == 3 and all(type(t) is int and t > 0 for t in grid),
                 'invalid original image grid')
        _require(_digest(binding.get('media_sha256')) and _digest(binding.get('encoding_sha256')),
                 'invalid original media/encoding identity')
        objects = image['objects']
        _require(isinstance(objects, list) and objects, 'empty or invalid full-label object list')
        object_ids = []
        for obj in objects:
            _require(isinstance(obj, dict) and set(obj) == set(full.OBJECT_FIELDS), 'full-label object schema changed')
            ann = obj['coco_ann_id']
            _require(type(ann) is int and (image_id, ann) not in annotation_keys, 'invalid or duplicate annotation ID')
            annotation_keys.add((image_id, ann))
            object_ids.append(ann)
            box = obj['bbox_2d']
            _require(isinstance(box, list) and len(box) == 4
                     and all(type(v) is int and 0 <= v <= 999 for v in box)
                     and box[0] < box[2] and box[1] < box[3], 'invalid full-label norm1000 box')
            _require(type(obj['category_id']) is int and isinstance(obj['desc'], str) and obj['desc']
                     and obj['category_name'] == obj['desc'], 'invalid full-label category')
            _require(categories.setdefault(obj['category_id'], obj['desc']) == obj['desc'],
                     'inconsistent full-label category name')
        _require(object_ids == binding.get('object_ids'), 'full annotation IDs/order differ from original manifest')
    _require(len(annotation_keys) == LABEL_COUNT, 'full-label input must contain all570 annotations')
    return images, manifest


def load_inputs(labels_path=FULL_LABEL_PATH, manifest_path=INPUT_MANIFEST_PATH):
    """Qualify input identity once at preparation; never hash a model/source tree."""
    _require(p.digest(labels_path) == FULL_LABEL_SHA256, 'frozen full-label bytes changed')
    _require(p.digest(manifest_path) == INPUT_MANIFEST_SHA256, 'original full-label manifest bytes changed')
    images, manifest = validate_inputs(p.load(labels_path), p.load(manifest_path))
    _require(manifest['sources'].get(str(p.POLICY)) == p.digest(p.POLICY), 'bound detection prompt policy changed')
    from PIL import Image
    for image in images:
        path = Path(image['image_path'])
        _require(path.is_file() and p.digest(path) == image['image_sha256'],
                 f"full-label image bytes changed: {image['image_id']}")
        with Image.open(path) as pixels:
            _require(pixels.size == (image['width'], image['height']), 'full-label image dimensions changed')
    return images, manifest


def request_records(images, manifest):
    """Generation inputs in original image order, sufficient for maintained native_request."""
    validate_inputs(images, manifest)
    return [dict(image_id=image['image_id'], cohort=image['cohort'],
                 image_path=image['image_path'], image_sha256=image['image_sha256'],
                 width=image['width'], height=image['height'], crop=[0, 0, image['width'], image['height']],
                 view_scale=1, request_id=f"rule-stability:input:{image['image_id']}",
                 prompt_token_ids=list(binding['prompt_token_ids']), image_grid_thw=list(binding['image_grid_thw']),
                 media_sha256=binding['media_sha256'], encoding_sha256=binding['encoding_sha256'])
            for image, binding in zip(images, manifest['images'], strict=True)]


def full_label_sequence(image, q):
    """Actual full-label encoding; object-owned atoms exclude terminal EOS."""
    from src.data.examples import RawExample, RawObject, ImageRef, SourceProvenance
    objects = sorted(image['objects'], key=lambda obj: (obj['bbox_2d'][0], obj['bbox_2d'][1], obj['coco_ann_id']))
    row_ids = tuple('full:' + str(obj['coco_ann_id']) for obj in objects)
    _require(len(row_ids) == len(set(row_ids)) and row_ids, 'full positive owner IDs must be unique and nonempty')
    raw = RawExample(str(image['image_id']),
        ImageRef(image['image_path'], Path(image['image_path']), image['width'], image['height'], {}),
        tuple(RawObject(oid, obj['desc'], tuple(obj['bbox_2d']), {})
              for oid, obj in zip(row_ids, objects, strict=True)), {},
        SourceProvenance(FULL_LABEL_PATH, 1, FULL_LABEL_SHA256, 'full570_positive_labels'))
    encoded, sequence = p.encode(raw, set(row_ids), q)
    _require(tuple(dict.fromkeys(atom.object_id for atom in sequence.atoms)) == row_ids,
             'encoded full-label owner order changed')
    _require(all(atom.object_id in row_ids and atom.token_type != 'eos' for atom in sequence.atoms),
             'terminal EOS entered full positive targets')
    return encoded, sequence, row_ids


def normalized_positive_objective(normalized_logits, sequence, row_ids, vocab, positions):
    """Use supplied policy-normalized logits in maintained CE+.1 type+.01 order row mean."""
    return retained.retained_objective(normalized_logits, sequence, row_ids, vocab, positions)


def balanced_layout(records, previous_records=None, version=0):
    """Maintained8-rank LPT assignment, at most3 images/rank and all18 images."""
    ids = [row['image_id'] for row in records]
    prompt_lengths = {row['image_id']: len(row['prompt_token_ids']) for row in records}
    return rollout.build_assignment(ids, prompt_lengths, previous_records, version)


def image_backward_scale(world_size=rollout.WORLD_SIZE):
    """Scale each image loss before DDP averaging: (world_size/18)*sum_local(image_losses)."""
    _require(type(world_size) is int and world_size > 0, 'world size must be a positive integer')
    return world_size / IMAGE_COUNT
