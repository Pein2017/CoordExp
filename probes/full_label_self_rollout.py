"""CPU preparation and frozen annotation accounting for the full-label fit."""
from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
from pathlib import Path

from probes import iterative_positive as p
from probes import rollout_row_credit as r

UNIT = Path(__file__).resolve().parents[1] / 'research/experiments/2026-10-02-full-label-self-rollout-fit'
INPUT_DIR = UNIT / 'inputs'
TRUTH = r.TRUTH
RETAINED = r.ROOT / 'cpu-04/retained-10.json'
INPUTS = r.ROOT / 'retained-sft-01/inputs.json'
ENCODINGS = r.ROOT / 'retained-sft-01/encodings.json'
POLICY = p.POLICY
FULL_FIELDS = ('cohort', 'height', 'image_id', 'image_path', 'image_sha256', 'objects', 'width')
OBJECT_FIELDS = ('bbox_2d', 'category_id', 'category_name', 'coco_ann_id', 'desc')
FULL_SHA256 = '4984b1b2819976471c1ada8e4697dca026604795fa8877bf75101c0bb7bfd56f'
RETAINED_SHA256 = 'ff9767850dfdc3c21c4ee714767d1d8e1281ecf096673b2dc1f4e4094f546dba'


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def decoder_runtime_identity() -> dict:
    distribution = metadata.distribution('vllm')
    if distribution.version != '0.29.0+cu129':
        raise RuntimeError(f"unsupported vLLM distribution version: {distribution.version}")
    paths = (Path(distribution.locate_file('vllm/v1/sample/sampler.py')),
             Path(distribution.locate_file('vllm/sampling_params.py')))
    if any(not path.is_file() for path in paths):
        raise RuntimeError('installed vLLM sampler sources are incomplete')
    return {'distribution': 'vllm', 'version': distribution.version,
            'source_sha256': {str(path): sha(path) for path in paths}}


def _load(path: Path):
    return json.loads(Path(path).read_text())


def _write_new(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        stream.write(p.canonical(value) + '\n')


def validate_inputs(full: list[dict], retained: list[dict], capture: list[dict]) -> None:
    assert len(full) == 18 and sum(len(image['objects']) for image in full) == 570
    assert len(retained) == 18 and sum(len(image['objects']) for image in retained) == 513
    assert len(capture) == 18
    ids = [image['image_id'] for image in full]
    assert all(type(image_id) is int for image_id in ids) and len(set(ids)) == 18
    assert set(image['image_id'] for image in retained) == set(ids)
    assert set(image['image_id'] for image in capture) == set(ids)
    assert len({image['image_id'] for image in retained}) == len({image['image_id'] for image in capture}) == 18
    full_by_id = {image['image_id']: image for image in full}
    capture_by_id = {image['image_id']: image for image in capture}
    retained_by_id = {image['image_id']: image for image in retained}
    categories = {}
    retained_total = 0
    for image in full:
        assert set(image) in (set(FULL_FIELDS), set(FULL_FIELDS) | {'hidden_objects'}), 'truth source fields changed'
        assert image['cohort'] in ('human13', 'refined5')
        assert type(image['width']) is int and type(image['height']) is int and min(image['width'], image['height']) > 0
        assert Path(image['image_path']).is_file()
        assert sha(Path(image['image_path'])) == image['image_sha256'], f"image bytes changed: {image['image_id']}"
        anns = [obj['coco_ann_id'] for obj in image['objects']]
        assert all(type(ann) is int for ann in anns) and len(anns) == len(set(anns))
        for obj in image['objects']:
            assert set(obj) == set(OBJECT_FIELDS)
            box = obj['bbox_2d']
            assert len(box) == 4 and all(type(v) is int and 0 <= v <= 999 for v in box)
            assert box[0] < box[2] and box[1] < box[3]
            assert type(obj['category_id']) is int and obj['category_name'] and obj['desc']
            assert obj['category_name'] == obj['desc']
            assert categories.setdefault(obj['category_id'], obj['category_name']) == obj['category_name']
        old = retained_by_id[image['image_id']]
        assert set(old) == set(FULL_FIELDS) and old['objects'] == [obj for obj in image['objects'] if obj['coco_ann_id'] in {x['coco_ann_id'] for x in old['objects']}]
        assert all(obj in image['objects'] for obj in old['objects'])
        retained_total += len(old['objects'])
        seen_capture = capture_by_id[image['image_id']]
        assert seen_capture['image_path'] == image['image_path'] and seen_capture['image_sha256'] == image['image_sha256']
        assert seen_capture['width'] == image['width'] and seen_capture['height'] == image['height']
    assert retained_total == 513
    missing = {(image['image_id'], obj['coco_ann_id']) for image in full for obj in image['objects']}
    retained_keys = {(image['image_id'], obj['coco_ann_id']) for image in retained for obj in image['objects']}
    assert len(missing - retained_keys) == 57 and retained_keys <= missing
    assert set(full_by_id) == set(capture_by_id)


def prepare_inputs() -> tuple[Path, Path]:
    inputs_path, manifest_path = INPUT_DIR / 'full-labels.json', INPUT_DIR / 'manifest.json'
    if inputs_path.exists() or manifest_path.exists():
        if inputs_path.exists() and manifest_path.exists():
            verify_inputs()
            return inputs_path, manifest_path
        raise FileExistsError('partial full-label input snapshot; preserve and inspect it before continuing')
    assert sha(TRUTH) == FULL_SHA256 and sha(RETAINED) == RETAINED_SHA256
    full, retained, capture, encodings = map(_load, (TRUTH, RETAINED, INPUTS, ENCODINGS))
    validate_inputs(full, retained, capture)
    assert len(encodings) == 18 and len({row['image_id'] for row in encodings}) == 18
    assert {row['image_id'] for row in encodings} == {row['image_id'] for row in full}
    # Keep only trusted source annotations and the original non-hidden image metadata.
    labels = [{key: image[key] for key in FULL_FIELDS} for image in full]
    _write_new(inputs_path, labels)
    capture_by_id = {image['image_id']: image for image in capture}
    encodings_by_id = {image['image_id']: image for image in encodings}
    retained_by_id = {image['image_id']: image for image in retained}
    manifest = {
        'schema': 'full-label-self-rollout-inputs-v1',
        'sources': {str(path): sha(path) for path in (TRUTH, RETAINED, INPUTS, ENCODINGS, POLICY)},
        'full_labels': {'path': str(inputs_path.relative_to(UNIT)), 'sha256': sha(inputs_path), 'images': 18, 'objects': 570},
        'retained': {'sha256': sha(RETAINED), 'images': 18, 'objects': 513, 'excluded_objects': 57},
        'images': [{
            'image_id': image['image_id'], 'cohort': image['cohort'],
            'object_ids': [obj['coco_ann_id'] for obj in image['objects']],
            'retained_object_ids': [obj['coco_ann_id'] for obj in retained_by_id[image['image_id']]['objects']],
            'image_sha256': image['image_sha256'], 'media_sha256': capture_by_id[image['image_id']]['media_sha256'],
            'prompt_token_ids': capture_by_id[image['image_id']]['prompt_token_ids'], 'image_grid_thw': capture_by_id[image['image_id']]['image_grid_thw'],
            'encoding_sha256': hashlib.sha256(p.canonical(encodings_by_id[image['image_id']]).encode()).hexdigest(),
        } for index, image in enumerate(full)],
        'limitations': 'Full annotation accounting only; annotation-unmatched predictions are not verified physical false positives.',
    }
    manifest['full_labels']['source_sha256'] = sha(TRUTH)
    _write_new(manifest_path, manifest)
    return inputs_path, manifest_path


def verify_inputs(unit: Path = UNIT) -> tuple[list[dict], dict]:
    """Revalidate the immutable snapshot and its original source bindings."""
    labels_path, manifest_path = unit / 'inputs/full-labels.json', unit / 'inputs/manifest.json'
    manifest = _load(manifest_path)
    assert manifest['schema'] == 'full-label-self-rollout-inputs-v1'
    assert manifest['sources'][str(TRUTH)] == FULL_SHA256
    assert manifest['sources'][str(RETAINED)] == RETAINED_SHA256
    assert sha(labels_path) == manifest['full_labels']['sha256']
    assert manifest['full_labels']['images'] == 18 and manifest['full_labels']['objects'] == 570
    for path, expected in manifest['sources'].items():
        assert sha(Path(path)) == expected, path
    labels = _load(labels_path)
    source = _load(TRUTH)
    expected_labels = [{key: image[key] for key in FULL_FIELDS} for image in source]
    assert labels == expected_labels, 'snapshot differs from the full-label source whitelist'
    retained = _load(RETAINED)
    capture = _load(INPUTS)
    validate_inputs(labels, retained, capture)
    expected_ids = [image['image_id'] for image in labels]
    image_bindings = manifest['images']
    assert [row['image_id'] for row in image_bindings] == expected_ids
    encodings = _load(ENCODINGS)
    assert len(encodings) == 18 and len({row['image_id'] for row in encodings}) == 18
    assert {row['image_id'] for row in encodings} == set(expected_ids)
    retained_by_id = {image['image_id']: image for image in retained}
    capture_by_id = {image['image_id']: image for image in capture}
    encodings_by_id = {image['image_id']: image for image in encodings}
    for image, binding in zip(labels, image_bindings):
        image_id = image['image_id']
        assert binding['object_ids'] == [obj['coco_ann_id'] for obj in image['objects']]
        assert binding['retained_object_ids'] == [obj['coco_ann_id'] for obj in retained_by_id[image_id]['objects']]
        assert binding['media_sha256'] == capture_by_id[image_id]['media_sha256']
        assert binding['prompt_token_ids'] == capture_by_id[image_id]['prompt_token_ids']
        assert binding['image_grid_thw'] == capture_by_id[image_id]['image_grid_thw']
        assert binding['encoding_sha256'] == hashlib.sha256(p.canonical(encodings_by_id[image_id]).encode()).hexdigest()
    return labels, manifest


def _f1(tp: int, fp: int, fn: int) -> float:
    return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0


def evaluate_versions(images: list[dict], frozen: dict[str, list[dict]]) -> dict:
    """Compare frozen outputs by class-agnostic IoU assignment then exact category."""
    expected = [image['image_id'] for image in images]
    assert len(expected) == 18 and sum(len(image['objects']) for image in images) == 570
    assert len(set(expected)) == 18 and 'zero' in frozen
    scored = {}
    for version, records in frozen.items():
        assert len(records) == 18 and {row['image_id'] for row in records} == set(expected)
        rows = r.assess_outputs(images, [], records)
        assert len(rows) == 18
        matched = {}
        outputs = {}
        for row, image in zip(rows, images):
            tp_ids = row['ids']['category']['retained']
            geometric_ids = row['ids']['raw']['retained']
            valid_count = row['burdens']['valid_rows']
            fp = valid_count - len(tp_ids)
            fn = len(image['objects']) - len(tp_ids)
            assert len(tp_ids) <= len(geometric_ids) <= len(image['objects'])
            annotation_ids = [obj['coco_ann_id'] for obj in image['objects']]
            outputs[image['image_id']] = {
                'cohort': image['cohort'], 'tp': len(tp_ids), 'fp_annotation': fp, 'fn': fn,
                'f1': _f1(len(tp_ids), fp, fn), 'denominator': len(image['objects']),
                'annotation_ids': annotation_ids,
                'missed_annotation_ids': sorted(set(annotation_ids) - set(tp_ids)),
                'annotation_unmatched_predictions': row['burdens']['unmatched'],
                'literal_valid_repeats': row['burdens']['literal_valid_repeats'],
                'literal_complete_repeats': row['burdens']['literal_complete_repeats'],
                'near_repeat_occurrence_pairs': row['burdens']['near_repeat_occurrence_pairs'],
                'category_disagreements': row['burdens']['category_disagreements'],
                'invalid_geometry': row['burdens']['geometry_invalid'], 'malformed': row['burdens']['malformed'],
                'eos': row['burdens']['eos'], 'caps': row['burdens']['caps'],
                'category_correct_ids': sorted(tp_ids), 'geometric_match_ids': sorted(geometric_ids),
            }
            matched[image['image_id']] = set(tp_ids)
        totals = {key: sum(row[key] for row in outputs.values()) for key in ('tp', 'fp_annotation', 'fn')}
        burden_keys = ('valid_rows', 'literal_valid_repeats', 'literal_complete_repeats',
                       'near_repeat_occurrence_pairs', 'geometry_invalid', 'malformed',
                       'generated_tokens', 'eos', 'caps', 'unmatched', 'category_disagreements')
        totals.update(denominator=570, f1=_f1(totals['tp'], totals['fp_annotation'], totals['fn']),
                      **{key: sum(row['burdens'][key] for row in rows) for key in burden_keys})
        totals['Nvalid'] = totals['valid_rows']
        scored[version] = {'images': outputs, 'totals': totals, '_matched': matched}

    base = scored['zero']['_matched']
    transitions = {}
    for version, result in scored.items():
        per_image = {}
        gained, lost = set(), set()
        for image_id in expected:
            before, after = base[image_id], result['_matched'][image_id]
            image_gained, image_lost = after - before, before - after
            gained.update((image_id, ann) for ann in image_gained)
            lost.update((image_id, ann) for ann in image_lost)
            per_image[image_id] = {'retained': sorted(before & after), 'gained': sorted(image_gained), 'lost': sorted(image_lost)}
        preserved = {(image_id, ann) for image_id in expected for ann in base[image_id] & result['_matched'][image_id]}
        transitions[version] = {'per_image': per_image,
                                'retained': sum(len(base[i] & result['_matched'][i]) for i in expected),
                                'preserved': sorted([list(key) for key in preserved]),
                                'preserved_count': len(preserved),
                                'gained': sorted([list(key) for key in gained]), 'lost': sorted([list(key) for key in lost]),
                                'gained_count': len(gained), 'lost_count': len(lost)}
        transitions[version]['lost_baseline_count'] = len(lost)
        transitions[version]['recovered_again'] = []
        transitions[version]['recovered_again_count'] = 0
    ordered = sorted(scored, key=lambda name: (0 if name == 'zero' else int(name) if name.isdigit() else 10**9, name))
    previous = {image_id: set(base[image_id]) for image_id in expected}
    lost_since_coverage = set()
    for version in ordered:
        if version == 'zero':
            continue
        current = scored[version]['_matched']
        now = {(image_id, ann) for image_id in expected for ann in current[image_id]}
        was = {(image_id, ann) for image_id in expected for ann in previous[image_id]}
        lost_since_coverage.update(was - now)
        recovered = now & lost_since_coverage
        lost_since_coverage.difference_update(recovered)
        transitions[version]['recovered_again'] = sorted([list(key) for key in recovered])
        transitions[version]['recovered_again_count'] = len(recovered)
        previous = {image_id: set(current[image_id]) for image_id in expected}
    for position, version in enumerate(ordered[1:], start=1):
        before_name = ordered[position - 1]
        before = scored[before_name]['_matched']
        after = scored[version]['_matched']
        adjacent_rows = {}
        adjacent_gained, adjacent_lost, adjacent_preserved = set(), set(), set()
        for image_id in expected:
            gained_now = after[image_id] - before[image_id]
            lost_now = before[image_id] - after[image_id]
            preserved_now = before[image_id] & after[image_id]
            adjacent_gained.update((image_id, ann) for ann in gained_now)
            adjacent_lost.update((image_id, ann) for ann in lost_now)
            adjacent_preserved.update((image_id, ann) for ann in preserved_now)
            adjacent_rows[image_id] = {'gained': sorted(gained_now), 'lost': sorted(lost_now),
                                       'preserved': sorted(preserved_now), 'gained_count': len(gained_now),
                                       'lost_count': len(lost_now), 'preserved_count': len(preserved_now)}
        transitions[version]['adjacent'] = {
            'from_version': before_name, 'per_image': adjacent_rows,
            'gained': sorted([list(key) for key in adjacent_gained]),
            'lost': sorted([list(key) for key in adjacent_lost]),
            'preserved': sorted([list(key) for key in adjacent_preserved]),
            'gained_count': len(adjacent_gained), 'lost_count': len(adjacent_lost),
            'preserved_count': len(adjacent_preserved),
        }
    tie = (4134, 294005)
    transitions['image_4134_annotation_294005'] = {
        version: {'zero_correct': tie[1] in base[tie[0]], 'version_correct': tie[1] in result['_matched'][tie[0]],
                  'zero_to_version': ('retained' if tie[1] in base[tie[0]] and tie[1] in result['_matched'][tie[0]] else
                                      'gained' if tie[1] not in base[tie[0]] and tie[1] in result['_matched'][tie[0]] else
                                      'lost' if tie[1] in base[tie[0]] and tie[1] not in result['_matched'][tie[0]] else 'missed')}
        for version, result in scored.items()
    }
    for result in scored.values():
        result.pop('_matched')
    return {'schema': 'full-label-self-rollout-metrics-v1', 'metric': 'annotation valid-row F1@IoU.5',
            'matcher': 'class_agnostic_cardinality_first_iou_gte_0.5_one_to_one_then_exact_description',
            'f1_formula': '2*TP/(570+Nvalid); Nvalid includes every strict valid parsed row including duplicates',
            'aggregation': 'micro_over_570_annotations', 'iou_threshold': 0.5, 'denominator_annotation_ids': sorted(
                [[image['image_id'], obj['coco_ann_id']] for image in images for obj in image['objects']]),
            'scored': scored,
            'transitions': transitions,
            'limitations': 'Annotation TP/FP/FN use class-agnostic one-to-one IoU assignment at 0.5 then exact description equality. Annotation-unmatched predictions and category disagreements are annotation errors, not verified physical false positives; no confidence scores or mAP are defined.'}


def prepare_qualification(unit: Path, root: Path, qualification_run: Path, observation_run: Path, checkpoint: Path) -> Path:
    from src.artifacts.git_identity import capture_source_identity
    from probes import online_row_credit as o

    input_path, manifest_path = unit / 'inputs/full-labels.json', unit / 'inputs/manifest.json'
    _, manifest = verify_inputs(unit)
    assert sha(input_path) == manifest['full_labels']['sha256']
    payload_path = checkpoint / 'inference_payload_manifest.json'
    anchor_sha = sha(payload_path)
    o.verify_anchor_payload(checkpoint, anchor_sha)
    training_path = input_path
    training_sha = sha(training_path)
    source = capture_source_identity(o.source_paths(full_label_region=True))
    recipe = o.full_label_recipe(checkpoint, anchor_sha, training_path, training_sha)
    qualification = {
        'schema': 'full-label-self-rollout-qualification-v1', 'unit': str(unit),
        'runs': {'qualification': str(qualification_run), 'observation': str(observation_run)},
        'correction': recipe,
        'input_manifest': {'path': str(manifest_path), 'sha256': sha(manifest_path)},
        'sha256': {str(path): sha(path) for path in (INPUTS, training_path, POLICY, manifest_path)},
        'source': source, 'rollout_backend': 'vllm', 'schema_geometry': True,
        'decoder_runtime_identity': decoder_runtime_identity(),
        'execution': {'profiles': [{'arm': 'treatment', 'microbatch': 1, 'activation_checkpointing': False}]},
        'pairs': {'2': {'treatment': str(qualification_run)}, '16': {'treatment': str(observation_run)}},
        'metric_definition': {'assignment': 'class_agnostic_one_to_one_iou_0.5_then_exact_description',
                              'tp': 'unique category-correct annotation matches',
                              'fp_annotation': 'all valid predictions minus tp, including duplicate rows',
                              'fn': '570 full-label annotations minus tp', 'f1': '2*tp/(2*tp+fp_annotation+fn)',
                              'physical_false_positive_claim': False},
    }
    out = root / 'qualification.json'
    _write_new(out, qualification)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('prepare-inputs')
    q = sub.add_parser('qualify')
    q.add_argument('--unit', type=Path, required=True)
    q.add_argument('--root', type=Path, required=True)
    q.add_argument('--qualification-run', type=Path, required=True)
    q.add_argument('--observation-run', type=Path, required=True)
    q.add_argument('--checkpoint', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare-inputs':
        prepare_inputs()
    else:
        prepare_qualification(args.unit, args.root, args.qualification_run, args.observation_run, args.checkpoint)


if __name__ == '__main__':
    main()
