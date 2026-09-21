"""Retained-data confidence screen only; no classifier fitting or GPU execution."""
import json,math,hashlib
from collections import Counter,defaultdict
from pathlib import Path

from src.artifacts import publish_json_exclusive

BASE=Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration')
OUT=BASE/'2026-09-10-unmatched-proxy-screen'
RUN=BASE/'2026-09-09-round1-greedy-realization/cold/source256-rloo-round1-train256-natural-v1'
AUDIT=BASE/'2026-09-09-fp-visual-distribution'
FEATURES=('coord_geomean','coord_probability_mean','coord_min','description_geomean','coord_description_geomean','native_score8')


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def quantiles(values):
    values=sorted(values)
    def q(frac):
        pos=(len(values)-1)*frac;a=int(pos);b=min(a+1,len(values)-1)
        return values[a]+(values[b]-values[a])*(pos-a)
    return {'n':len(values),'min':values[0],'q25':q(.25),'median':q(.5),'q75':q(.75),'max':values[-1]} if values else {'n':0}


def auc(pos,neg):
    return sum(1 if a>b else .5 if a==b else 0 for a in pos for b in neg)/(len(pos)*len(neg))


def main():
    sources={}
    def rows(path):
        sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        return read_rows(path)
    reviews=json.loads((AUDIT/'final-v2/reviewed-cases.json').read_text())
    sources[str(AUDIT/'final-v2/reviewed-cases.json')]=hashlib.sha256((AUDIT/'final-v2/reviewed-cases.json').read_bytes()).hexdigest()
    inventory={x['case_id']:x for x in rows(AUDIT/'inventory.jsonl')}
    traces=defaultdict(dict)
    for t in rows(RUN/'pred_token_trace.jsonl'):
        if t['trace_type']=='generated_token':
            assert t['generated_step_index'] not in traces[t['row_id']]
            traces[t['row_id']][t['generated_step_index']]=t
    features={}
    for row in rows(RUN/'gt_vs_pred_scored.jsonl'):
        image_id=str(int(row['row_id'].rsplit('_',1)[1]))
        trace=traces[row['row_id']]
        for j,pred in enumerate(row['pred']):
            proof=pred['pred_score_source']
            steps=proof['generated_step_indices']; texts=proof['token_text'];logs=proof['selected_logprobs']
            assert len(steps)==len(texts)==len(logs)==8
            assert all(trace[k]['token_id']==token and trace[k]['logprob']==lp for k,token,lp in zip(steps,proof['token_ids'],logs))
            coords=[lp for text,lp in zip(texts,logs) if text.startswith('<|coord_')]
            assert len(coords)==4
            a=steps[texts.index('<|object_ref_start|>')];b=steps[texts.index('<|object_ref_end|>')]
            desc=[trace[k]['logprob'] for k in range(a+1,b)]
            assert desc and all(math.isfinite(x) and x<=0 for x in coords+desc)
            assert math.isclose(pred['score'],math.exp(sum(logs)/8),rel_tol=1e-12)
            key=f'{image_id}:p{j}'
            if key in inventory: assert list(pred['bbox'])==inventory[key]['bbox']
            features[key]={'case_id':key,'category':pred['description'],'bbox':pred['bbox'],
                'coord_geomean':math.exp(sum(coords)/4),
                'coord_probability_mean':sum(math.exp(x) for x in coords)/4,
                'coord_min':math.exp(min(coords)),
                'description_geomean':math.exp(sum(desc)/len(desc)),
                'coord_description_geomean':math.exp(sum(coords+desc)/len(coords+desc)),
                'native_score8':pred['score'],'description_token_count':len(desc),
                'group':inventory[key]['stratum'] if key in inventory else 'matched_TP'}
    assert len(features)==2356 and len(inventory)==1094
    joined=[]
    for r in reviews:
        f=features[r['case_id']]
        assert r['source_sample']['stratum']!='strict_repeat' and f['bbox']==r['source_sample']['bbox']
        clean=(r['primary_reason']=='unlabeled_real_instance' and r['category']=='correct' and r['geometry']=='acceptable')
        defective=r['primary_reason'] in ('category_error','localization_error','multi_instance_box','duplicate_prediction')
        assert not(clean and defective)
        joined.append(dict(f,primary_reason=r['primary_reason'],review_confidence=r['confidence'],
                           descriptive_binary='clean_extra' if clean else 'defective_current_row' if defective else 'not_binary_labeled'))
    pos=[x for x in joined if x['descriptive_binary']=='clean_extra']
    neg=[x for x in joined if x['descriptive_binary']=='defective_current_row']
    assert len(joined)==54 and len(pos)==13 and len(neg)==20
    ranking={}
    for feature in FEATURES:
        ranking[feature]={'auc_high_predicts_clean_extra':auc([x[feature] for x in pos],[x[feature] for x in neg]),
            'clean_extra':quantiles([x[feature] for x in pos]),'defective_current_row':quantiles([x[feature] for x in neg]),
            'fixed_thresholds':[{ 'threshold':cut,'clean_pass':sum(x[feature]>=cut for x in pos),
                'defective_pass':sum(x[feature]>=cut for x in neg),
                'gray_pass':sum(x[feature]>=cut for x in joined if x['descriptive_binary']=='not_binary_labeled')}
                for cut in (.3,.5,.7)]}
    label_distributions={label:{f:quantiles([x[f] for x in joined if x['primary_reason']==label]) for f in FEATURES}
                         for label in sorted({x['primary_reason'] for x in joined})}
    population={group:{f:quantiles([x[f] for x in features.values() if x['group']==group]) for f in FEATURES}
                for group in sorted({x['group'] for x in features.values()})}
    result={'schema_version':'unmatched_confidence_screen.v1','scope':'Retained Source256 round1 confidence signals versus54 provisional visual labels; no fit or deployment validation',
        'source_files':sources,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'score_definitions':{'coord_geomean':'exp(mean(logp of4 coordinate tokens))','coord_probability_mean':'mean(exp(logp) of4 coordinate tokens)',
          'coord_min':'min(exp(logp) of4 coordinate tokens)','description_geomean':'exp(mean(logp of actual description tokens))',
          'coord_description_geomean':'exp(mean(logp over coordinates and description tokens))','native_score8':'Current artifact score:4 coordinates+4schema tokens; excludes description/category'},
        'binary_denominator':{'clean_extra':13,'defective_current_row':20,'not_binary_labeled':21},
        'ranking':ranking,'visual_label_feature_distributions':label_distributions,'population_feature_distributions':population,
        'limitations':'Descriptive reuse of an outcome-stratified audit with non-gold model labels; no heldout calibration, no learned threshold, no hallucination negatives, no training-label promotion. Not_binary_labeled entries are not negatives.',
        'gpu_forwards':0,'classifier_training_steps':0,'dataset_label_changes':0}
    publish_json_exclusive(OUT/'features.json',joined)
    publish_json_exclusive(OUT/'summary.json',result)
    print(json.dumps({'binary_denominator':result['binary_denominator'],'ranking':ranking,'population_coord_geomean':{k:v['coord_geomean'] for k,v in population.items()}},indent=2))


if __name__=='__main__':main()
