# Bounded CPU diagnosis query

Run the single Python block from the research-probes checkout with `PYTHONDONTWRITEBYTECODE=1`.
It reuses the native parser and global owner scorer, verifies aggregate arithmetic,
and writes only this diagnosis's JSON plus the three authorized visual projections.
The selected images illustrate different observed error-count patterns; they do
not estimate the prevalence of physically false predictions.

```python
from pathlib import Path
from collections import Counter
import json
import sys

sys.dont_write_bytecode = True
REPO = Path('/data/CoordExp/.worktrees/research-probes')
sys.path.insert(0,str(REPO))
from probes.dora_owner_learning.candidate_opportunity import digest,file_hash,require,score
from probes.dora_owner_learning.entrance_ce_eval import aggregate_scores,owner_change
from probes.dora_owner_learning.reward_rows import _gt_objects,_pred_objects,_bbox,_pixel_box
from probes.source_rweak_row_cross.run import native_record
from src.data.geometry import iou_xyxy

ROOT = Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-10-selective-owner-learning-autonomous')
OUT = ROOT/'dev112-diagnosis'
DEV = ROOT/'soft-preservation-10/dev112'
EVAL = ROOT/'soft-preservation-10/evaluation/execution'
TH = ('50','60','80')
VISUAL_IDS = (226097,167952,460339)

def read(p):return json.loads(p.read_text())
def publish(p,d):
    p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():require(read(p)==d,'diagnosis would overwrite different data')
    else:
        with p.open('x') as f:json.dump(d,f,indent=2,sort_keys=True);f.write('\n')

manifest,consumer,reduction = [read(DEV/name) for name in ('manifest.json','consumer.json','reduction.json')]
frozen = {r['example_id']:r for r in manifest['records']}
require(len(frozen)==len(consumer)==112 and {r['example_id'] for r in consumer}==set(frozen),'112-image coverage')
require(all(r['manifest_sha256']==digest(manifest) for r in consumer),'native consumer manifest identity')

def characterize(parsed,scored):
    projection = []
    geometry = []
    repeat = {}
    for j,obj in enumerate(parsed['pred']):
        box = _pixel_box(_bbox(obj))
        if box is not None:
            repeat[j] = any(iou_xyxy(box,old)>.95 for old in geometry)
            geometry.append(box)
        if _pred_objects(dict(parsed,pred=[obj]))[0]:projection.append(j)
    require(sum(repeat.values())==scored['strict_repeats'],'strict-repeat recount')
    result = dict(category_predictions=dict(Counter(p['description'] for p in parsed['pred'])),
                  category_strict_repeats=dict(Counter(parsed['pred'][j]['description'] for j,b in repeat.items() if b)),thresholds={})
    for t in TH:
        matched = {projection[m['pred_index']] for m in scored[t]['matches']}
        fps = [j for j in projection if j not in matched]
        repeat_fp = [j for j in fps if repeat[j]]
        nonrepeat_fp = [j for j in fps if not repeat[j]]
        require(len(fps)==scored[t]['fp'],'FP projection/count identity')
        result['thresholds'][t] = dict(repeat_fp=len(repeat_fp),nonrepeat_fp=len(nonrepeat_fp),
            repeated_tp=sum(repeat[j] for j in matched),
            category_fp=dict(Counter(parsed['pred'][j]['description'] for j in fps)),
            category_repeat_fp=dict(Counter(parsed['pred'][j]['description'] for j in repeat_fp)),
            category_nonrepeat_fp=dict(Counter(parsed['pred'][j]['description'] for j in nonrepeat_fp)))
    return result

per_image = []
losses = {t:[] for t in TH}
reassignments = {t:{'gained':[],'lost':[]} for t in TH}
for row in consumer:
    old = frozen[row['example_id']]
    for parsed,text,stop,ids,saved in [(old['baseline'],old['baseline']['raw_decode_text'],old['baseline']['decode_stop_reason'],old['baseline_ids'],old['baseline_score']),
                                     (row['parsed'],row['text'],row['stop_reason'],row['action_ids'],row['score'])]:
        require(native_record(text,old['case'],old['baseline'],stop)==parsed,'native parser replay')
        require(score(parsed,seed=-1,length=len(ids),stop=stop)==saved,'native global-score replay')
    s,c = old['baseline_score'],row['score']
    a,b = characterize(old['baseline'],s),characterize(row['parsed'],c)
    d = dict(image_id=old['image_id'],example_id=row['example_id'],source=s,candidate=c,
        delta={k:c[k]-s[k] for k in ('prediction_count','strict_repeats','parser_drops','complete_token_length')},
        owner_changes=row['owner_changes'],matching_reassignment=row['matching_reassignment'],
        source_characterization=a,candidate_characterization=b,metrics={})
    for t in TH:
        require(owner_change(c[t]['owners'],s[t]['owners'])==row['owner_changes'][t],'owner-set recount')
        d['metrics'][t] = {k:c[t][k]-s[t][k] for k in ('tp','fp','fn','f1')}
        d['metrics'][t].update({k:b['thresholds'][t][k]-a['thresholds'][t][k] for k in ('repeat_fp','nonrepeat_fp','repeated_tp')})
        for kind in ('gained','lost'):
            for owner in row['matching_reassignment'][t][kind]:reassignments[t][kind].append(dict(image_id=old['image_id'],owner=owner))
        for owner in row['owner_changes'][t]['lost']:
            gt_index=next(i for i,g in enumerate(old['baseline']['gt']) if str(g['object_id'])==owner)
            category,gt_box=_gt_objects(old['baseline'],row_id=row['example_id'])[gt_index]
            best={}
            for label,parsed in [('source',old['baseline']),('candidate',row['parsed'])]:
                best[label]=max((iou_xyxy(gt_box,box) for cat,box in _pred_objects(parsed)[0] if cat==category),default=0.)
            losses[t].append(dict(image_id=old['image_id'],owner=owner,category=category,best_category_consistent_iou=best,
                candidate_still_directly_supported=best['candidate']>=int(t)/100))
    if old['image_id'] in VISUAL_IDS:
        d['literal_patterns']={}
        for label,parsed in [('source',old['baseline']),('candidate',row['parsed'])]:
            d['literal_patterns'][label]=dict(
                top_x1=[dict(bin=value,count=count) for value,count in Counter(p['coord_bins'][0] for p in parsed['pred']).most_common(8)],
                top_exact_rows=[dict(description=key[0],bins=list(key[1]),count=count) for key,count in
                    Counter((p['description'],tuple(p['coord_bins'])) for p in parsed['pred']).most_common(8)])
    per_image.append(d)

source=aggregate_scores([r['source'] for r in per_image])
candidate=aggregate_scores([r['candidate'] for r in per_image])
require(source==reduction['source'] and candidate==reduction['candidate'],'aggregate source/candidate recount')
require(candidate['50']['fp']-source['50']['fp']==123 and candidate['50']['tp']-source['50']['tp']==18 and
        candidate['strict_repeats']-source['strict_repeats']==41,'requested primary delta verification')
for t in TH:
    sums={kind:sum(len(r['owner_changes'][t][kind]) for r in per_image) for kind in ('gained','lost','retained')}
    require(sums==reduction['owner_changes'][t],'aggregate owner gain/loss recount')

def concentration(metric):
    value = (lambda r:r['metrics']['50'][metric]) if metric in ('fp','repeat_fp','nonrepeat_fp') else (
        (lambda r:r['delta']['strict_repeats']) if metric=='repeats' else (lambda r:len(r['owner_changes']['50']['lost'])))
    ranked=sorted(per_image,key=lambda r:(-value(r),r['image_id']))
    pos=sum(max(value(r),0) for r in ranked);neg=sum(min(value(r),0) for r in ranked)
    return dict(gross_positive=pos,gross_negative=neg,net=pos+neg,
        positive_images=sum(value(r)>0 for r in ranked),unchanged_images=sum(value(r)==0 for r in ranked),
        negative_images=sum(value(r)<0 for r in ranked),
        top=[dict(image_id=r['image_id'],contribution=value(r)) for r in ranked[:10]],
        top_shares={str(k):dict(sum=sum(max(value(r),0) for r in ranked[:k]),
            fraction_of_gross_positive=sum(max(value(r),0) for r in ranked[:k])/pos if pos else None,
            fraction_of_net=sum(max(value(r),0) for r in ranked[:k])/(pos+neg) if pos+neg else None) for k in (1,3,5,10)})

def group_summary(group):
    return dict(images=len(group),image_ids=[r['image_id'] for r in group],
        delta={k:sum(r['delta'][k] for r in group) for k in ('prediction_count','strict_repeats','parser_drops','complete_token_length')},
        tp50=sum(r['metrics']['50']['tp'] for r in group),fp50=sum(r['metrics']['50']['fp'] for r in group),
        gross_positive_fp50=sum(max(r['metrics']['50']['fp'],0) for r in group),
        repeat_fp50=sum(r['metrics']['50']['repeat_fp'] for r in group),nonrepeat_fp50=sum(r['metrics']['50']['nonrepeat_fp'] for r in group))

drop_groups={name:group_summary([r for r in per_image if pred(r['delta']['parser_drops'])]) for name,pred in
    [('improved',lambda x:x<0),('unchanged',lambda x:x==0),('worse',lambda x:x>0)]}
ranked_fp=sorted(per_image,key=lambda r:(-r['metrics']['50']['fp'],r['image_id']))
exclusion={}
for k in (1,3,5,10):
    ids={r['image_id'] for r in ranked_fp[:k]};subset=[r for r in per_image if r['image_id'] not in ids]
    ss,cc=[aggregate_scores([r[label] for r in subset]) for label in ('source','candidate')]
    exclusion[str(k)]=dict(excluded_ids=sorted(ids),images=len(subset),source50=ss['50'],candidate50=cc['50'],
                          f1_delta=cc['50']['f1']-ss['50']['f1'],scope='Descriptive same-image exclusion, not a corrected evaluation.')
categories={}
for metric in ('category_predictions','category_strict_repeats','category_fp','category_repeat_fp','category_nonrepeat_fp'):
    totals={}
    for label in ('source','candidate'):
        c=Counter()
        for row in per_image:
            char=row[label+'_characterization'];c.update(char[metric] if metric in char else char['thresholds']['50'][metric])
        totals[label]=dict(c)
    names=set(totals['source'])|set(totals['candidate'])
    totals['delta']={k:totals['candidate'].get(k,0)-totals['source'].get(k,0) for k in sorted(names)}
    categories[metric]=totals

first18=read(EVAL/'reduction.json')
result=dict(schema='dev112_concentration_diagnosis.v1',status='candidate_read_only_diagnostic',images=112,
    inputs={str(p):file_hash(p) for p in [DEV/'manifest.json',DEV/'consumer.json',DEV/'reduction.json',EVAL/'reduction.json',Path(__file__) ]} if '__file__' in globals() and Path(__file__).is_file() else
           {str(p):file_hash(p) for p in [DEV/'manifest.json',DEV/'consumer.json',DEV/'reduction.json',EVAL/'reduction.json']},
    checks=dict(native_parse_replays=224,native_score_replays=224,aggregate_exact=True,owner_gain_loss_exact=True,
                model_loads=0,model_forwards=0),source=source,candidate=candidate,owner_changes=reduction['owner_changes'],
    concentration={m:concentration(m) for m in ('fp','repeats','lost','repeat_fp','nonrepeat_fp')},
    parser_drop_groups=drop_groups,category_totals=categories,owner_loss_details=losses,matching_reassignment=reassignments,
    exclusion_sensitivity=exclusion,per_image=per_image,visual_ids=list(VISUAL_IDS),
    first18_context={split:{key:data[key] for key in ('source','candidate','owner_changes')} for split,data in first18['splits'].items()},
    boundaries=['Annotation-relative FP is not a physical false-object verdict.',
                'Net prediction increase is not an identity count of newly discovered physical instances.',
                'Strict later pixel-IoU>.95 repeats and repeated-FP counts are distinct because matching can select a later repeat.',
                'Gross-positive concentration and net concentration have different denominators.',
                'Drop-improved image membership does not causally match a former malformed span to a new valid object.',
                'Visual overlay greedy matching is illustrative; all reported owner metrics use the native global assignment.'])
publish(OUT/'analysis.json',result)

from src.vis import render_prediction_comparison
visual=OUT/'visual'
ordered=[f'coco2017_train_{iid:012d}' for iid in VISUAL_IDS]
by_id={r['example_id']:r for r in consumer}
for label in ('source','candidate'):
    root=visual/label;root.mkdir(parents=True,exist_ok=True)
    payload=[frozen[eid]['baseline'] if label=='source' else by_id[eid]['parsed'] for eid in ordered]
    for name in ('gt_vs_pred.jsonl','gt_vs_pred_scored.jsonl'):
        path=root/name
        if path.exists():require([json.loads(s) for s in path.read_text().splitlines()]==payload,'visual input changed')
        else:
            with path.open('x') as stream:
                for row in payload:stream.write(json.dumps(row)+'\n')
    publish(root/'projection.json',dict(scope='Unchanged selected native rows; no likelihood fabrication or geometry/GT edits.',
        source=str(DEV/('manifest.json' if label=='source' else 'consumer.json')),source_sha256=file_hash(DEV/('manifest.json' if label=='source' else 'consumer.json')),row_ids=ordered))
if not (visual/'comparison/manifest.json').exists():
    rendered=render_prediction_comparison(visual/'source',visual/'candidate',visual/'comparison',
        left_label='Source step2444',right_label='KL10 selective',row_ids=ordered,duplicate_iou_threshold=.95)
    require(len(rendered.image_paths)==3 and all(p.is_file() for p in rendered.image_paths),'exact3 comparison renders')
else:
    rendered=read(visual/'comparison/manifest.json')
    require(len(rendered['items'])==3 and [r['row_id'] for r in rendered['items']]==ordered,'render readback coverage')
print(json.dumps(dict(status='complete',analysis_sha256=file_hash(OUT/'analysis.json'),checks=result['checks'],
    concentration=result['concentration'],parser_drop_groups=result['parser_drop_groups'],
    matching_reassignment=reassignments,exclusion_sensitivity=exclusion)))
```
