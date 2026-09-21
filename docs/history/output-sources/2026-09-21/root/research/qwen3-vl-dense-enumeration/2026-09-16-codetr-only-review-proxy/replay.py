"""Retrospective CPU-only contrast; no detector/VLM inference or annotation writes."""
import collections
import hashlib
import json
from pathlib import Path

SOURCE = Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-10-autonomous-unmatched-evaluator')
OUT = Path(__file__).resolve().parent / 'replay-v1'


def read(path):
    return json.loads(path.read_text())


def binding(path):
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def summarize(rows, arm, reference):
    supported = [r for r in rows if r[arm]]
    counts = collections.Counter(r[reference] for r in supported)
    total_clean = sum(r[reference] == 'clean' for r in rows)
    definite = counts['clean'] + counts['defective']
    return {'population': len(rows), 'supported': len(supported),
            'clean': counts['clean'], 'defective': counts['defective'], 'gray': counts['gray'],
            'coverage': len(supported)/len(rows),
            'clean_retention': counts['clean']/total_clean,
            'definite_support_precision': counts['clean']/definite if definite else None,
            'verified_fraction_among_all_supported': counts['clean']/len(supported) if supported else None}


def main():
    paths = {k: SOURCE/v for k,v in {
        'spec': 'selected-candidate-v1.json',
        'candidates': 'holdout-v1/candidates.jsonl',
        'detector': 'codetr-context-holdout-v1/decisions.json',
        'combined': 'holdout-evaluation-v2/decisions-before-unblinding.json',
        'original_reference': 'holdout-evaluation-v1/cases.json',
        'adjudicated_reference': 'holdout-evaluation-v2/cases.json',
        'historical_summary': 'holdout-evaluation-v2/summary.json',
    }.items()}
    spec = read(paths['spec'])
    assert spec['detector_score_min'] == .5 and spec['candidate_detector_iou_min'] == .75
    candidates = [json.loads(l) for l in paths['candidates'].read_text().splitlines()]
    detector = {r['case_id']: r for r in read(paths['detector'])}
    assert len(candidates) == len(detector) == len({r['image_id'] for r in candidates}) == 64
    assert {r['case_id'] for r in candidates} == set(detector)
    # Freeze the detector-only predictions before opening reference or VLM decisions.
    predictions = []
    for candidate in candidates:
        d = detector[candidate['case_id']]
        s = d['selected']
        support = bool(s and s['score'] >= .5 and s['category'] == candidate['category'] and d['candidate_iou'] >= .75)
        assert support == (d['decision'] == 'accept')
        predictions.append({'case_id': candidate['case_id'], 'image_id': candidate['image_id'],
                            'detector_only_support': support, 'candidate_iou': d['candidate_iou'],
                            'category_conflict_flag': d['category_conflict_flag'],
                            'training_admitted': False})
    OUT.mkdir(exist_ok=False)
    (OUT/'predictions.json').write_text(json.dumps(predictions, indent=2)+'\n')
    combined = {r['case_id']: r for r in read(paths['combined'])}
    original = {r['case_id']: r for r in read(paths['original_reference'])}
    revised = {r['case_id']: r for r in read(paths['adjudicated_reference'])}
    assert set(detector) == set(combined) == set(original) == set(revised)
    rows = []
    for p in predictions:
        case = p['case_id']
        rows.append({**p, 'combined_support': combined[case]['decision']=='accept_candidate',
                     'original_reference': original[case]['reference_class'],
                     'adjudicated_reference': revised[case]['reference_class'],
                     'reference_axes': {k: revised[case]['reference'][k] for k in ['entity','category','geometry','confidence']}})
    assert all(not r['combined_support'] or r['detector_only_support'] for r in rows)
    summary = {'status': 'retrospective_replay_completed', 'new_model_calls': 0,
               'rule': {'detector_score_min': .5, 'same_category_iou_min': .75},
               'results': {ref: {arm: summarize(rows,arm,ref) for arm in ['detector_only_support','combined_support']}
                           for ref in ['original_reference','adjudicated_reference']},
               'incremental_cases': [r for r in rows if r['detector_only_support'] and not r['combined_support']],
               'input_bindings': {k: binding(v) for k,v in paths.items()}, 'producer': binding(Path(__file__).resolve()),
               'limits': ['Already unblinded historical64-image panel; not fresh validation.',
                          'Provisional visual references; not human gold or new training labels.',
                          'No automatic teacher admission and no exhaustive owner/cross-prediction uniqueness proof.']}
    historical=read(paths['historical_summary'])
    paired=summary['results']['adjudicated_reference']['combined_support']
    assert [paired[k] for k in ['clean','defective','gray']]==[historical[k] for k in ['accepted_clean','accepted_defective','accepted_gray']]
    (OUT/'cases.json').write_text(json.dumps(rows,indent=2)+'\n')
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({'results': summary['results'], 'incremental_cases': summary['incremental_cases']}))


if __name__ == '__main__':
    main()
