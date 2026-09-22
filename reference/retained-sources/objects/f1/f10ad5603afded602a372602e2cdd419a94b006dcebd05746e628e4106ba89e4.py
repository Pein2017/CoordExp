"""Decision and raw-row diagnostics; unchanged native384 metric denominator."""
from collections import Counter
import json
from pathlib import Path
from probes.dora_owner_learning.candidate_opportunity import file_hash
from probes.dora_owner_learning.route_access import publish

OUT=Path(__file__).parent


def row_counts(parsed,ids):
    return dict(raw_object_starts=ids.count(151646),valid_rows=len(parsed['pred']),
        parser_drops=parsed['dropped_prediction_count'],
        drop_reasons=dict(Counter(d['reason'] for d in parsed['dropped_predictions'])),tokens=len(ids))


def build():
    inputs=OUT/'inputs.json'; consumer=OUT/'evaluation/consumer.json'; reduction=OUT/'evaluation/reduction.json'
    m=json.loads(inputs.read_text());r=json.loads(reduction.read_text());cc={x['image_id']:x for x in json.loads(consumer.read_text())}
    before={x['image_id']:x for x in m['eval_records']};u=r['panels']['union384'];a,b=u['anchor'],u['candidate']
    criteria=dict(strict_repeats_decrease=b['strict_repeats']<a['strict_repeats'],aggregate_TP50_not_lower=b['50']['tp']>=a['50']['tp'],
        aggregate_F1_not_lower=b['50']['f1']>=a['50']['f1'],protected7_retained=all(x['50'] for x in r['protected_targets'].values()),
        no_new_cap_image=not r['new_cap_images'],invalid_prediction_count_not_higher=b['invalid_predictions']<=a['invalid_predictions'],
        parser_drop_count_not_higher=b['parser_drops']<=a['parser_drops'])
    raw={}
    for name,parsed,ids in [('anchor',[x['stable_parsed'] for x in before.values()],[x['stable_ids'] for x in before.values()]),
                            ('candidate',[x['parsed'] for x in cc.values()],[x['action_ids'] for x in cc.values()])]:
        counts=[row_counts(p,t) for p,t in zip(parsed,ids)];reasons=Counter()
        for c in counts:reasons.update(c['drop_reasons'])
        raw[name]={k:sum(c[k] for c in counts) for k in ('raw_object_starts','valid_rows','parser_drops','tokens')}
        raw[name]['drop_reasons']=dict(reasons)
    online=[]
    for x in r['per_image']:
        if x['split']!='online8':continue
        iid=x['image_id'];old,new=before[iid],cc[iid]
        online.append(dict(image_id=iid,anchor=row_counts(old['stable_parsed'],old['stable_ids']),candidate=row_counts(new['parsed'],new['action_ids']),
            strict_repeats=[x['anchor']['strict_repeats'],x['candidate']['strict_repeats']],TP50=[x['anchor']['50']['tp'],x['candidate']['50']['tp']],
            gained=x['owner_changes']['50']['gained'],lost=x['owner_changes']['50']['lost']))
    return dict(schema='stable50_geometric_dedup.interpretation.v1',status='not_promoted_fixed_dose_joint_failure' if not all(criteria.values()) else 'requires_visual_decision',
        criteria=criteria,failed_criteria=[k for k,v in criteria.items() if not v],raw_row_diagnostics=raw,online=online,
        source_files={str(p):file_hash(p) for p in (inputs,consumer,reduction)},
        interpretation_boundary='Geometry-invalid rows are parser drops, not valid predictions. Lower strict-repeat/FP counts alone do not prove fewer generated row attempts. IoU50 losses are annotation-relative, not automatically lost physical instances.')


if __name__=='__main__':
    result=build();path=OUT/'interpretation.json'
    if path.exists():assert result==json.loads(path.read_text())
    else:publish(path,result)
    print(json.dumps({k:result[k] for k in ['status','criteria','failed_criteria','raw_row_diagnostics']}))
