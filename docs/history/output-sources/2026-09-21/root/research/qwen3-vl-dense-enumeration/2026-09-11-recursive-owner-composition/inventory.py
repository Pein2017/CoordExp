"""Lead recomputation of candidate rank in native pixel geometry."""
import json
from pathlib import Path
from collections import Counter
from probes.dora_owner_learning.candidate_opportunity import file_hash
from probes.dora_owner_learning.route_access import publish
from probes.dora_owner_learning.reward_rows import _gt_objects
from src.data.geometry import iou_xyxy

OUT=Path(__file__).parent
ANCHOR=OUT.parent/'2026-09-10-selective-owner-learning-autonomous/positive7-support50-81'
PROTECTED={368,7116,252411,465695,529411,538814,540567}


def build():
    manifest_path=ANCHOR/'evaluation/manifest.json'; consumer_path=ANCHOR/'evaluation/consumer.json'
    m=json.loads(manifest_path.read_text()); cc={r['example_id']:r for r in json.loads(consumer_path.read_text())}
    eligible=[]; counts=Counter()
    for f in m['records']:
        if f['split']=='dev128' or f['image_id'] in PROTECTED: continue
        counts['nonprotected_train']+=1; r=cc[f['example_id']]; s=r['score']; parsed=r['parsed']
        if s['50']['fn']<2: continue
        counts['missing_ge2']+=1
        if any(s[k] for k in ('cap','strict_repeats','parser_drops','invalid_predictions')) or r['stop_reason']!='im_end': continue
        counts['clean_missing_ge2']+=1
        if not 3<=len(parsed['gt'])<=12 or len(r['action_ids'])>256: continue
        counts['moderate_short_missing_ge2']+=1
        if s['50']['fn']!=2:
            counts['ge3_nonexecuted']+=1; continue
        counts['eligible_exact2']+=1
        pixel=_gt_objects(parsed,row_id=f['example_id']); width,height=parsed['image_width'],parsed['image_height']
        miss=[]
        for i,g in enumerate(parsed['gt']):
            if str(g['object_id']) in s['50']['owners']: continue
            box=pixel[i][1]; area=(box[2]-box[0])*(box[3]-box[1])/(width*height)
            miss.append(dict(owner=str(g['object_id']),category=pixel[i][0],bbox_norm999=g['bbox'],bbox_pixels=list(box),area_fraction=area))
        assert len(miss)==2
        miss.sort(key=lambda g:(*g['bbox_norm999'],int(g['owner'])))
        key=[-min(g['area_fraction'] for g in miss),iou_xyxy(miss[0]['bbox_pixels'],miss[1]['bbox_pixels']),len(r['action_ids']),f['image_id']]
        eligible.append(dict(example_id=f['example_id'],image_id=f['image_id'],split=f['split'],
            image_path=parsed['image_path'],image_width=width,image_height=height,gt_count=len(parsed['gt']),
            missing_owner_ids=[g['owner'] for g in miss],preserved_owner_ids=s['50']['owners'],missing_owners=miss,
            stable_original_generated_token_count=len(r['action_ids']),sort_key=key))
    eligible.sort(key=lambda c:c['sort_key']); selected=eligible[:24]
    for i,c in enumerate(selected): c['selection_rank']=i+1
    counts['selected_exact2']=len(selected)
    counts.update({'selected_'+k:v for k,v in Counter(c['split'] for c in selected).items()})
    paths=[manifest_path,consumer_path,Path(__file__),OUT/'candidate-inventory.luna-high-rejected.json']
    return dict(schema='recursive_owner_composition.candidate_inventory.v2',status='lead_recomputed_before_GPU',
        generated_from={str(p):file_hash(p) for p in paths},counts=dict(counts),
        criteria=dict(primary_missing=2,GT_count=[3,12],max_action_tokens=256,native_EOS=True,
            zero_burden=['cap','strict_repeats','parser_drops','invalid_predictions'],excluded_image_ids=sorted(PROTECTED),
            selection_limit=24,sort='descending minimum missing-owner pixel-area fraction, ascending missing-pair pixel IoU, ascending action length, numeric image id',
            geometry='Native _gt_objects converts norm999 GT bins to pixel boxes; area divided by pixel image area.'),
        eligible_image_ids=[c['image_id'] for c in eligible],candidates=selected)


if __name__=='__main__':
    result=build(); publish(OUT/'candidate-inventory.json',result)
    old=json.loads((OUT/'candidate-inventory.luna-high-rejected.json').read_text())
    a={int(c['image_id']) for c in old['candidates']}; b={c['image_id'] for c in result['candidates']}
    print(json.dumps(dict(counts=result['counts'],selected=[c['image_id'] for c in result['candidates']],
        removed_after_unit_fix=sorted(a-b),added_after_unit_fix=sorted(b-a))))
