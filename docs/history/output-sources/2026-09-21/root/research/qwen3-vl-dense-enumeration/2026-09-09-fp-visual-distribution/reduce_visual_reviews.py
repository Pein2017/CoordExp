"""Validate fixed visual reviews and report exact versus exploratory counts."""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from src.artifacts import publish_json_exclusive

ROOT=Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-09-fp-visual-distribution')
REASONS={'duplicate_prediction','unlabeled_real_instance','localization_error','category_error',
         'multi_instance_box','annotation_extent_or_grouping','hallucinated_entity','uncertain'}
ENUMS={'primary_reason':REASONS,
       'entity':{'real_single','real_multiple','no_supported_entity','uncertain'},
       'category':{'correct','incorrect','uncertain'},
       'geometry':{'acceptable','too_tight','too_loose','shifted','multi_instance','uncertain'},
       'confidence':{'high','medium','low'}}


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(output_dir, overrides_path):
    scope=json.loads((ROOT/'active-review-scope.json').read_text())
    sample={x['case_id']:x for x in rows(ROOT/'sample.jsonl')}
    active=set(scope['active_case_ids'])
    assert len(active)==54 and all(sample[k]['stratum']!='strict_repeat' for k in active)
    reviews=[]
    source_hashes={str(ROOT/'active-review-scope.json'):sha(ROOT/'active-review-scope.json'),
                   str(ROOT/'sample.jsonl'):sha(ROOT/'sample.jsonl')}
    overrides={}
    if overrides_path is not None:
        override_rows=rows(overrides_path)
        overrides={x['case_id']:x for x in override_rows}
        assert len(overrides)==len(override_rows) and set(overrides)<=active
        source_hashes[str(overrides_path)]=sha(overrides_path)
    for batch in ('root','a','b','c'):
        path=ROOT/'reviews'/f'{batch}.jsonl'
        records=rows(path)
        expected={x['case_id'] for x in rows(ROOT/'batches'/f'{batch}.jsonl') if x['case_id'] in active}
        assert len(records)==len(expected) and {x['case_id'] for x in records}==expected,(batch,'coverage')
        source_hashes[str(path)]=sha(path)
        for record in records:
            if record['case_id'] in overrides:
                override=overrides[record['case_id']]
                record=dict(record,initial_review=dict(record),lead_override_reason=override['reason'],
                            **override['replacement'])
            item=sample[record['case_id']]
            for field,allowed in ENUMS.items(): assert record[field] in allowed,(record['case_id'],field)
            assert record['evidence'].strip() and record['reviewer']
            viewed=set(record['viewed_paths'])
            assert item['crop_path'] in viewed
            assert item['image_path'] in viewed or item['overview_path'] in viewed
            assert all(Path(p).is_file() for p in viewed)
            reviews.append(dict(record,source_sample=item))
    assert len(reviews)==54 and {x['case_id'] for x in reviews}==active
    by_stratum={}
    weighted=defaultdict(float)
    for stratum in ('same_category_near_GT','other_category_high_overlap','weak_GT_relation'):
        records=[x for x in reviews if x['source_sample']['stratum']==stratum]
        assert records
        n=records[0]['source_sample']['stratum_sample_size']
        N=records[0]['source_sample']['population_size']
        assert len(records)==n
        by_stratum[stratum]={'population':N,'reviewed':n,
            'primary_counts':dict(Counter(x['primary_reason'] for x in records)),
            'entity_counts':dict(Counter(x['entity'] for x in records)),
            'confidence_counts':dict(Counter(x['confidence'] for x in records))}
        for record in records: weighted[record['primary_reason']]+=record['source_sample']['sample_weight']
    assert abs(sum(weighted.values())-610)<1e-8
    summary={'schema_version':'fp_visual_review_reduction.v1','status':'candidate_for_lead_acceptance',
        'source_files':source_hashes,'code_sha256':sha(Path(__file__)),
        'lead_override_case_ids':sorted(overrides),
        'exact_population':{'fp':1094,'automatic_strict_repeat_fp':484,'nonrepeat_fp':610,
                            'fp_in_four_capped_images':511,'capped_and_repeat_counts_overlap':True},
        'visual_sample':{'predictions':54,'images':len({x['source_sample']['example_id'] for x in reviews}),
            'primary_counts':dict(Counter(x['primary_reason'] for x in reviews)),
            'entity_counts':dict(Counter(x['entity'] for x in reviews)),
            'category_counts':dict(Counter(x['category'] for x in reviews)),
            'geometry_counts':dict(Counter(x['geometry'] for x in reviews)),
            'confidence_counts':dict(Counter(x['confidence'] for x in reviews))},
        'strata':by_stratum,
        'exploratory_design_weighted_nonrepeat_counts':dict(weighted),
        'uncertainty':'Visual judgments are non-certified model-review evidence. Raw sample proportions are not population proportions because stratum sampling fractions differ. Weighted point estimates are exploratory and do not account for visual misclassification; no precise population rate or zero-hallucination claim follows.',
        'training_use':'Not ground truth; no dataset, reward or training change authorized by this audit.'}
    reviews.sort(key=lambda x:x['case_id'])
    output_dir.mkdir(parents=True,exist_ok=False)
    publish_json_exclusive(output_dir/'reviewed-cases.json',reviews)
    summary['reviewed_cases_sha256']=sha(output_dir/'reviewed-cases.json')
    publish_json_exclusive(output_dir/'review-summary.json',summary)
    print(json.dumps({k:summary[k] for k in ('exact_population','visual_sample','strata','exploratory_design_weighted_nonrepeat_counts')},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--overrides',type=Path)
    args=parser.parse_args()
    main(args.output_dir,args.overrides)
