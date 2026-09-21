"""Replayable CPU-only fresh128 repetition census. No physical duplicate claims."""
import collections
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAT = re.compile(r'<\|object_ref_start\|>(.*?)<\|object_ref_end\|><\|box_start\|>' + r'<\|coord_(\d+)\|>' * 4 + r'<\|box_end\|>')
ARMS = {'O': 'baseline', 'N': 'treated'}
SCREENSHOTS = ['99184', '84241', '114820', '313465', '495443', '483867', '542582']

def read(path):
    return json.loads(path.read_text())

def bind(path):
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

def cat(text):
    return text.strip().lower()

def iou(a, b):
    inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0

def new_cat():
    return collections.Counter()

def analyze(arm, overlay):
    rows = arm['complete_rows']
    preds = arm['valid_predictions']
    per_cat = collections.defaultdict(new_cat)
    literal_seen = set()
    native_seen = set()
    details = []
    strict = []
    for row in rows:
        c = cat(row['description'])
        box = row['box']
        valid = box[0] < box[2] and box[1] < box[3]
        key = (row['description'], tuple(box))
        kind = 'valid_bins' if valid else 'invalid_bins'
        per_cat[c]['complete_rows'] += 1
        per_cat[c][kind + '_rows'] += 1
        if key in literal_seen:
            per_cat[c]['literal_' + kind + '_repeats'] += 1
        literal_seen.add(key)
    for j, pred in enumerate(preds):
        c = cat(pred['description'])
        counts = per_cat[c]
        counts['native_valid_rows'] += 1
        key = (pred['description'], tuple(pred['coord_bins_1000']))
        literal = key in native_seen
        counts['literal_native_valid_repeats'] += int(literal)
        native_seen.add(key)
        row_detail = {'prediction_id': pred['prediction_id'], 'generated_order': pred['generated_order'], 'category': c, 'coord_bins': pred['coord_bins_1000'], 'literal_repeat': literal}
        for space, field in [('bins', 'coord_bins_1000'), ('pixel', 'bbox_pixel_xyxy')]:
            best_any = max(((iou(pred[field], prev[field]), k) for k, prev in enumerate(preds[:j])), default=(0.0, -1))
            best_same = max(((iou(pred[field], prev[field]), k) for k, prev in enumerate(preds[:j]) if cat(prev['description']) == c), default=(0.0, -1))
            for mode, best in [('any', best_any), ('same', best_same)]:
                flag = best[0] > .95
                counts[f'strict_{space}_{mode}_later_rows'] += int(flag)
                row_detail[f'strict_{space}_{mode}'] = flag
                if flag:
                    row_detail[f'earlier_{space}_{mode}'] = {'prediction_id': preds[best[1]]['prediction_id'], 'iou': best[0]}
            if space == 'bins' and best_any[0] > .95:
                strict.append({'prediction_id': pred['prediction_id'], 'generated_order': pred['generated_order'], 'best_earlier_iou': best_any[0]})
        if literal or any(row_detail[f'strict_{s}_{m}'] for s in ['bins', 'pixel'] for m in ['any', 'same']):
            details.append(row_detail)
    for run in arm['exact_runs']:
        c = cat(run['description'])
        per_cat[c]['longest_exact_run'] = max(per_cat[c]['longest_exact_run'], run['length'])
    candidate_later = collections.defaultdict(set)
    for pair in overlay['duplicate_candidates']:
        c = cat(pair['description'])
        per_cat[c]['overlay_iou30_pairs'] += 1
        candidate_later[c].add(pair['pred_b_index'])
    for c, later in candidate_later.items():
        per_cat[c]['overlay_iou30_later_rows'] = len(later)
    assert len(strict) == arm['burden']['strict_valid_repeats']
    assert (strict[0] if strict else None) == arm['first_strict_repeat']
    assert sum(c['literal_valid_bins_repeats'] + c['literal_invalid_bins_repeats'] for c in per_cat.values()) == arm['burden']['literal_repeats']
    assert sum(c['literal_invalid_bins_repeats'] for c in per_cat.values()) == arm['burden']['literal_invalid_repeats']
    return {'burden': arm['burden'], 'stop': arm['stop'], 'token_count': arm['token_count'], 'per_category': {c: dict(v) for c, v in sorted(per_cat.items())}, 'first_strict_repeat': arm['first_strict_repeat'], 'longest_exact_run': arm['longest_exact_run'], 'runs_length_ge2': [r for r in arm['exact_runs'] if r['length'] >= 2], 'repeat_rows': details, 'overlay': {k: overlay[k] for k in ['tp', 'fp', 'fn']}, 'overlay_iou30_pairs': len(overlay['duplicate_candidates']), 'overlay_iou30_examples': sorted(overlay['duplicate_candidates'], key=lambda p: -p['iou'])[:8]}

def main():
    saved = read(ROOT / 'result.json')
    images = {iid: image for iid, image in saved['images'].items() if image['cohort'] == 'fresh'}
    manifest_path = ROOT / 'user-comparison-v1/fresh/png/manifest.json'
    manifest = read(manifest_path)
    overlays = {str(int(item['row_id'].rsplit('_', 1)[1])): item for item in manifest['items']}
    assert len(images) == len(overlays) == 128
    raw_cache = {}
    output = {}
    gt_exposures = collections.defaultdict(set)
    for iid, image in images.items():
        overlay = overlays[iid]
        for gt in overlay['gt_objects']:
            gt_exposures[cat(gt['description'])].add(iid)
        output[iid] = {}
        for key, arm in ARMS.items():
            binding = image['raw_bindings'][key]
            path = Path(binding['path'])
            if str(path) not in raw_cache:
                assert bind(path)['sha256'] == binding['sha256']
                raw_cache[str(path)] = read(path)
            raw = next(r for r in raw_cache[str(path)]['rows'] if str(r['image_id']) == iid)
            complete = [{'row': j + 1, 'description': m[1], 'box': list(map(int, m.groups()[1:]))} for j, m in enumerate(PAT.finditer(raw['text']))]
            assert complete == image[arm]['complete_rows']
            output[iid][key] = analyze(image[arm], overlay['left' if key == 'O' else 'right']['match'])
    categories = sorted(set(gt_exposures) | {c for image in output.values() for arm in image.values() for c in arm['per_category']})
    cat_stats = {}
    for c in categories:
        union_ids = [iid for iid, image in output.items() if any(image[a]['per_category'].get(c, {}).get('native_valid_rows', 0) for a in ARMS)]
        cat_stats[c] = {'gt_image_exposure': len(gt_exposures[c]), 'union_prediction_image_exposure': len(union_ids)}
        for key in ARMS:
            items = [(iid, image[key]['per_category'].get(c, {})) for iid, image in output.items()]
            metrics = sorted({m for _, stats in items for m in stats})
            totals = {m: sum(stats.get(m, 0) for _, stats in items) for m in metrics if m != 'longest_exact_run'}
            totals['images_with_native_valid_rows'] = sum(stats.get('native_valid_rows', 0) > 0 for _, stats in items)
            for m in ['strict_bins_any_later_rows', 'strict_bins_same_later_rows', 'strict_pixel_same_later_rows', 'literal_valid_bins_repeats', 'literal_invalid_bins_repeats', 'overlay_iou30_pairs']:
                totals[m + '_image_count'] = sum(stats.get(m, 0) > 0 for _, stats in items)
                totals[m + '_image_ids'] = [iid for iid, stats in items if stats.get(m, 0) > 0]
            totals['longest_exact_run'] = max((stats.get('longest_exact_run', 0) for _, stats in items), default=0)
            totals['strict_bins_any_top_images'] = [{'image_id': iid, 'later_rows': stats.get('strict_bins_any_later_rows', 0), 'native_valid_rows': stats.get('native_valid_rows', 0), 'longest_exact_run': stats.get('longest_exact_run', 0)} for iid, stats in sorted(items, key=lambda t: -t[1].get('strict_bins_any_later_rows', 0)) if stats.get('strict_bins_any_later_rows', 0) > 0][:6]
            cat_stats[c][key] = totals
    aggregate = {}
    for key in ARMS:
        totals = collections.Counter()
        for image in output.values():
            a = image[key]
            for k, v in a['burden'].items():
                if isinstance(v, int): totals[k] += v
            totals['images_with_strict_repeats'] += a['burden']['strict_valid_repeats'] > 0
            totals['images_with_literal_repeats'] += a['burden']['literal_repeats'] > 0
            totals['images_with_literal_valid_repeats'] += a['burden']['literal_repeats'] > a['burden']['literal_invalid_repeats']
            totals['images_with_literal_invalid_repeats'] += a['burden']['literal_invalid_repeats'] > 0
            totals['overlay_iou30_pairs'] += a['overlay_iou30_pairs']
        aggregate[key] = dict(totals)
    paired = {}
    for metric in ['strict_valid_repeats', 'literal_repeats', 'literal_invalid_repeats']:
        paired[metric] = {direction: [{'image_id': iid, 'O': image['O']['burden'][metric], 'N': image['N']['burden'][metric]} for iid, image in output.items() if test(image['N']['burden'][metric] - image['O']['burden'][metric])] for direction, test in [('increased', lambda x: x > 0), ('decreased', lambda x: x < 0)]}
    result = {'scope': '128 sampled fresh images only; counts are geometric/token repeats, not human-confirmed same-physical-owner duplicates', 'definitions': {'literal': 'Repeated (case-sensitive description, exact four coord bins) beyond first in regex-complete rows; split bins-valid/invalid geometry', 'strict_saved': 'Each native valid later row counted once if ANY earlier category has bin-space IoU strictly > 0.95; exactly replayed against saved burden', 'strict_sensitivity': 'Also compute same-normalized-category and pixel-space IoU strictly > 0.95; attribution is later-row category', 'overlay': 'All same-normalized-category pixel-IoU >= 0.30 unordered pairs, including matched/matched; later-row count is candidate indicator only', 'image_denominators': 'Provide arm prediction exposure, O/N union prediction exposure, and annotated GT category exposure; category counts overlap across images', 'onset': 'generated_order/P indices zero-based; exact run start_row is one-based'}, 'sources': [bind(ROOT / 'result.json'), bind(manifest_path), bind(ROOT / 'cohort/cohort_manifest.json'), bind(Path(__file__))], 'raw_files_sha256_verified': len(raw_cache), 'saved_counters_replayed': True, 'aggregate': aggregate, 'paired': paired, 'categories': cat_stats, 'screenshots': {iid: output[iid] for iid in SCREENSHOTS}, 'images': output}
    target = ROOT / 'duplication-discussion/census.json'
    target.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'path': str(target), 'aggregate': aggregate, 'strict_changes': paired['strict_valid_repeats'], 'categories_with_strict': {c: {a: {'rows': d[a].get('native_valid_rows', 0), 'images': d[a]['images_with_native_valid_rows'], 'strict_rows': d[a].get('strict_bins_any_later_rows', 0), 'strict_images': d[a]['strict_bins_any_later_rows_image_count'], 'literal_invalid': d[a].get('literal_invalid_bins_repeats', 0)} for a in ARMS} for c, d in cat_stats.items() if any(d[a].get('strict_bins_any_later_rows', 0) for a in ARMS)}}, indent=2))

if __name__ == '__main__':
    main()
