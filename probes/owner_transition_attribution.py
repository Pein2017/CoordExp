"""CPU derivative of the frozen round04 evaluator; no model loading or rematching."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import resource
import subprocess
import time

from probes import rollout_row_credit as rows
from probes.hidden_human_recovery import canonical, candidates, load, write
from src.eval.saved_rows import iou_xyxy
from src.inference.token_text import character_span_to_token_interval

PREDECESSOR = Path('/data/CoordExp/.worktrees/greedy-prefix-native-01/outputs/research/physical-fn-recovery/2026-10-03/prefix-exposure-ranking-04')
OUT = Path(__file__).resolve().parents[1] / 'outputs/research/physical-fn-recovery/2026-10-03/owner-transition-attribution-05'
FROZEN = {
    'released-contract-01.json': '0fe3deac8f3eff7ec542686bdda8c7673aeee91b993bd3ca1628ac2233bf441b',
    'native-terminal-candidate-01.json': '78583948601bdb86b94e3d08d6086b82d452d0c8f3fd834cac235af375304080',
    'lead-acceptance-01.json': 'c93f0863496fde0f14404ca4af8daa2733f7c88f37bc405553131b4b21cba931',
}
ARMS = ('anchor', 'R-single', 'R-multiple')
CLASSES = ('eligible_same_category_not_credited', 'only_other_categories_at_threshold',
           'same_category_overlap_below_threshold', 'no_positive_same_category_overlap')
SEMANTICS = dict(threshold=.5, inclusive=True, assignment='class-agnostic maximum cardinality then maximum total round(IoU*1e9)',
    category_credit='exact description equality after assignment; no rematch', matching_pool='all strict-parser-valid rows including duplicates',
    geometry='full-image normalized bins without pixel rounding', classification_side='uncovered anchor for gain; uncovered endpoint for loss',
    classes=list(CLASSES), null_overlap='no same-description candidate', zero_overlap='same-description candidate exists with zero overlap',
    row_order='candidate generated_order is compressed; prediction-ID suffix is raw parser order',
    token_positions='zero-based original generated IDs; intervals are half-open; structural boundaries only, no lexical retokenization',
    causal_classification=False, annotation_unmatched_is_physical_negative=False)
SOURCE_PATHS = ['probes/owner_transition_attribution.py', 'tests/probes/test_owner_transition_attribution.py']


def require(ok, message):
    if not ok:
        raise ValueError(message)


def bound_inputs():
    """Read and hash each consumed file once, against its predecessor binding."""
    bindings, values = {}, {}

    def read(path, expected):
        path = Path(path)
        if str(path) not in values:
            payload = path.read_bytes()
            actual = hashlib.sha256(payload).hexdigest()
            require(actual == expected, f'input identity changed: {path}')
            bindings[str(path)] = actual
            values[str(path)] = json.loads(payload)
        require(bindings[str(path)] == expected, f'conflicting binding: {path}')
        return values[str(path)]

    c, terminal, accepted = [read(PREDECESSOR / name, sha) for name, sha in FROZEN.items()]
    require(accepted['status'] == 'lead-accepted', 'predecessor not accepted')
    saved = read(accepted['terminal']['path'], accepted['terminal']['sha256'])
    require(terminal['artifact_sha256'][accepted['terminal']['path']] == accepted['terminal']['sha256'], 'terminal binding differs')
    labels = read(c['label_path'], c['evidence_bindings'][c['label_path']])
    require(len(labels) == 18 and sum(len(r['objects']) for r in labels) == 570, 'denominator differs')
    label_indices = {r['image_id']: j for j, r in enumerate(labels)}
    require(len(label_indices) == 18 and set(label_indices) == set(c['images']), 'reference/image IDs differ')
    labels = [labels[label_indices[i]] for i in c['images']]
    tokenizer = read(Path(c['base_model']) / 'tokenizer.json', c['tokenizer_sha256'])
    special = {t['content']: t['id'] for t in tokenizer['added_tokens'] if t['special']}
    require([special[f'<|coord_{i}|>'] for i in range(1000)] == c['coordinate_ids'], 'coordinate identity differs')
    records = {}
    for arm in ARMS:
        records[arm] = []
        for image in c['images']:
            path = PREDECESSOR / f'package-01/native-{arm}/natural-{image}.json'
            value = read(path, terminal['artifact_sha256'][str(path)])
            raw = value['raw']
            require(raw['image_id'] == image and raw['crop'] == [0, 0, raw['width'], raw['height']], 'raw geometry differs')
            require(raw['generated_tokens'] == len(raw['token_ids']) and raw['token_ids'] == value['acquisition']['token_ids'], 'raw token identity differs')
            sealed = {k: raw[k] for k in ('producer', 'request_id', 'image_id', 'token_ids', 'text', 'prompt_token_ids', 'media_sha256', 'image_grid_thw', 'stop_reason')}
            require(hashlib.sha256(canonical(sealed).encode()).hexdigest() == raw['raw_identity'], 'raw seal differs')
            records[arm].append(dict(path=str(path), **value))
    return dict(contract=c, accepted=accepted, terminal=terminal, saved=saved, labels=labels,
                records=records, special=special, bindings=bindings, label_indices=label_indices)


def positioned_pool(raw, special):
    """Map existing parser evidence to saved structural IDs, without decoding."""
    parsed = rows.parse(raw)
    pool, _ = candidates([raw])
    by_order = {p['generated_order']: p for p in parsed.predictions}
    by_id = {ident: text for text, ident in special.items()}
    pattern = re.compile('|'.join(re.escape(s) for s in sorted(special, key=len, reverse=True)))
    markers = list(pattern.finditer(raw['text']))
    positions = [(i, by_id[t]) for i, t in enumerate(raw['token_ids']) if t in by_id]
    require([m.group() for m in markers] == [s for _, s in positions], 'structural token/text sequence differs')
    spans = [(-1, -1)] * len(raw['token_ids'])
    for m, (i, _) in zip(markers, positions, strict=True):
        spans[i] = (m.start(), m.end())
    for item in pool:
        order = int(item['prediction_id'].rsplit(':p', 1)[1])
        row = by_order[order]
        interval = character_span_to_token_interval(row['char_start'], row['char_end'], text=raw['text'], token_spans=spans)
        coordinates = [character_span_to_token_interval(s['char_start'], s['char_end'], text=raw['text'], token_spans=spans)[0] for s in row['coord_token_spans']]
        require(len(coordinates) == 4 and coordinates == list(range(coordinates[0], coordinates[0] + 4)), 'coordinate positions differ')
        require([raw['token_ids'][i] for i in coordinates] == [special[f'<|coord_{b}|>'] for b in row['coord_bins']], 'coordinate token/bin differs')
        item.update(raw_parser_order=order, char_interval=[row['char_start'], row['char_end']],
                    token_interval=list(interval), coordinate_positions=coordinates, raw_span_sha256=row['raw_span_sha256'])
    return pool


def side_witness(image, obj, record, measurement, pool):
    matches = measurement['matches']
    by_prediction = {m['prediction_id']: m for m in matches}
    by_owner = {o['coco_ann_id']: o for o in image['objects']}
    overlaps = [(p, iou_xyxy(obj['bbox_2d'], p['coord_bins_1000'])) for p in pool]
    same = [(p, v) for p, v in overlaps if p['description'] == obj['desc']]
    eligible_same = [(p, v) for p, v in same if v >= .5]
    eligible_any = [(p, v) for p, v in overlaps if v >= .5]

    def witness(pair):
        p, overlap = pair
        match = by_prediction.get(p['prediction_id'])
        recipient = by_owner[int(match['reference_owner_id'])] if match else None
        return dict(**p, iou_to_reference=overlap, assigned_to=None if match is None else dict(
            match, description=recipient['desc'], reference_box=recipient['bbox_2d'], category_credited=p['description'] == recipient['desc']))

    def best(pairs):
        return None if not pairs else witness(max(pairs, key=lambda x: x[1]))

    assigned = next((m for m in matches if m['reference_owner_id'] == str(obj['coco_ann_id'])), None)
    actual = None if assigned is None else witness(next(x for x in overlaps if x[0]['prediction_id'] == assigned['prediction_id']))
    covered = obj['coco_ann_id'] in measurement['ids']['category']['retained']
    return dict(raw_locator=record['path'] + '#/raw', raw_identity=record['raw']['raw_identity'],
        valid_candidate_count=len(pool), same_description_candidate_count=len(same),
        eligible_same_description_count=len(eligible_same), eligible_any_description_count=len(eligible_any),
        best_same_description=best(same), best_any_description=best(overlaps),
        eligible_same_description=[witness(x) for x in eligible_same],
        assignment_status='unassigned' if actual is None else 'assigned_same_description' if actual['description'] == obj['desc'] else 'assigned_different_description',
        actual_assignment=actual, raw_covered=obj['coco_ann_id'] in measurement['ids']['raw']['retained'], category_covered=covered)


def classify(side):
    if side['eligible_same_description_count']:
        return CLASSES[0]
    if side['eligible_any_description_count']:
        return CLASSES[1]
    best = side['best_same_description']
    if best is not None and 0 < best['iou_to_reference'] < .5:
        return CLASSES[2]
    return CLASSES[3]


def projection(inputs):
    c, labels, records, saved = (inputs[k] for k in ('contract', 'labels', 'records', 'saved'))
    measured, pools, totals = {}, {}, {}
    for arm in ARMS:
        measured[arm], pools[arm] = [], []
        for image, record, historical in zip(labels, records[arm], saved['natural'][arm], strict=True):
            value = rows.assess_outputs([image], [], [record['raw']])[0]
            require(value == record['measurement'] == historical, f'inherited evaluator mismatch: {arm}/{image["image_id"]}')
            measured[arm].append(value)
            pools[arm].append(positioned_pool(record['raw'], inputs['special']))
        values = measured[arm]
        totals[arm] = dict(image_count=len(values), raw_owners=sum(len(v['ids']['raw']['retained']) for v in values),
            category_owners=sum(len(v['ids']['category']['retained']) for v in values),
            burdens={k: sum(v['burdens'][k] for v in values) for k in values[0]['burdens']})
    require(totals == inputs['terminal']['natural_totals'] == inputs['accepted']['natural_totals'], 'accepted totals differ')
    require([totals[a]['category_owners'] for a in ARMS] == [274, 276, 268], 'coverage differs')
    entries, aggregates, transition_counts = [], {}, {}
    for arm, counts in [('R-single', (31, 29)), ('R-multiple', (24, 30))]:
        aggregates[arm], transition_counts[arm] = {}, {}
        historical = saved['natural_transitions'][arm]['images']
        for direction, expected in zip(('gain', 'loss'), counts, strict=True):
            counter = Counter()
            for j, image in enumerate(labels):
                before, after = measured['anchor'][j], measured[arm][j]
                require(historical[j]['image_id'] == image['image_id'], 'transition image order differs')
                for mode in ('raw', 'category'):
                    b, a = set(before['ids'][mode]['retained']), set(after['ids'][mode]['retained'])
                    actual = dict(gained=sorted(a-b), lost=sorted(b-a), retained=sorted(a & b))
                    require(actual == historical[j]['owners'][mode], 'accepted transition IDs differ')
                require({k: after['burdens'][k] - before['burdens'][k] for k in before['burdens']} == historical[j]['burden_deltas'], 'burden deltas differ')
                ids = historical[j]['owners']['category']['gained' if direction == 'gain' else 'lost']
                for obj in image['objects']:
                    if obj['coco_ann_id'] not in ids:
                        continue
                    sides = {a: side_witness(image, obj, records[a][j], measured[a][j], pools[a][j]) for a in ('anchor', arm)}
                    uncovered = 'anchor' if direction == 'gain' else arm
                    require(not sides[uncovered]['category_covered'] and sides[arm if direction == 'gain' else 'anchor']['category_covered'], 'transition credit differs')
                    classification = classify(sides[uncovered])
                    counter[classification] += 1
                    entries.append(dict(instance_id=f'{arm}/{image["image_id"]}/{obj["coco_ann_id"]}/{direction}',
                        arm=arm, direction=direction, image_id=image['image_id'], annotation_id=obj['coco_ann_id'],
                        reference=dict(description=obj['desc'], bbox_2d=obj['bbox_2d'], cohort=image['cohort'],
                            label_locator=c['label_path'] + f'#/{inputs["label_indices"][image["image_id"]]}/objects/{image["objects"].index(obj)}'),
                        uncovered_side=uncovered, classification=classification, sides=sides))
            require(sum(counter.values()) == expected, 'transition instance count differs')
            transition_counts[arm][direction] = expected
            aggregates[arm][direction] = {name: counter[name] for name in CLASSES}
    require(len(entries) == len({e['instance_id'] for e in entries}) == 114, '114 instance reconciliation failed')
    return dict(schema='owner-transition-attribution-v1', semantics=SEMANTICS, entries=entries, classes_by_arm_direction=aggregates,
        reconciliation=dict(records=54, images=18, annotations=570, instances=114, transition_counts=transition_counts,
            natural_totals=totals, all_saved_measurements_equal=True, exact_transition_id_sets_equal=True),
        provenance=dict(input_sha256=inputs['bindings'], saved_execution_source=c['source']['commit'],
            accepted_token_text_validation_reused=True, tokenizer_use='JSON structural ID lookup only; no tokenizer/processor/model loading'))


def consume(path, inputs):
    claimed = load(path)
    expected = projection(inputs)
    require({k: claimed.get(k) for k in expected} == expected and set(claimed) == set(expected) | {'analysis_source'},
            'attribution differs from bound raw records')
    source = claimed['analysis_source']
    require(source == analysis_source(source['commit']), 'analysis source revision/diff differs')
    return dict(status='candidate', instances=114, records=54, all_decision_fields_recomputed=True,
                attribution_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())


def analysis_source(commit=None):
    commit = commit or subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    require(re.fullmatch('[0-9a-f]{40}', commit) is not None, 'invalid source revision')
    require(subprocess.check_output(['git', 'ls-files', '--', *SOURCE_PATHS], text=True).splitlines() == SOURCE_PATHS, 'stage new source before binding revision/diff')
    return dict(cwd=str(Path.cwd()), commit=commit,
                diff=subprocess.check_output(['git', 'diff', commit, '--', *SOURCE_PATHS], text=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('build', 'readback'))
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    started = time.monotonic()
    inputs = bound_inputs()
    path = args.output / 'attribution.json'
    if args.command == 'build':
        value = projection(inputs)
        value['analysis_source'] = analysis_source()
        write(path, value)
    result = consume(path, inputs)
    result['resources'] = dict(wall_seconds=time.monotonic()-started, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                              output_bytes=path.stat().st_size)
    result['unexpected_resource_bound'] = result['resources']['wall_seconds'] > 300 or result['resources']['peak_rss_bytes'] > 8*1024**3
    print(canonical(result))


if __name__ == '__main__':
    main()
