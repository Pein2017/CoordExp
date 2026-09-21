import json,importlib.util
from pathlib import Path
R=Path(__file__).parent;p=json.loads((R/'panel.json').read_text());d=json.loads((R/'reduction.json').read_text());a=json.loads((R/'annotation-identity.json').read_text());ann=next(json.loads(x) for x in Path(a['working']['path']).read_text().splitlines() if json.loads(x)['image_id']==417044)
src=R.parent/'2026-09-16-endpoint-loop-natural-readout-norm/reduce.py';spec=importlib.util.spec_from_file_location('accepted_score',src);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
bank=[dict(image_id=417044,owner_id=str(o['coco_ann_id']),description=o['desc'],normalized_description=o['desc'].strip().lower(),reference_coord_bins_1000=o['bbox_2d']) for o in ann['objects']]
rows={}
for x in d['paths']:
 score=m.score(dict(text=x['row_text'],token_ids=x['row_token_ids'],stop='row_end'),p['group']['cases'][p['target']],bank)
 if x['branch']==0:physical=dict(classification='repeated_A',owner_id='-1693019979812657',coverage='A seeded in native row5',extent='small extent variants around same dominant donut; border included')
 elif x['branch']==52:physical=dict(classification='unvisited_B',owner_id='-8380415314849442',coverage='admitted B not covered in retained first five rows',extent='upper edge includes small surrounding context; dominant donut distinct from A')
 else:physical=dict(classification='physical_identity_HOLD',owner_id=None,coverage='unresolved',extent='small partial box at boundary of adjacent upper-left donuts/reflection; cannot certify dominant complete owner or previous coverage')
 rows[x['id']]=dict(scoring=score,physical=physical,overlay_index=next(z['index'] for z in json.loads((R/'physical/index.json').read_text()) if x['id'] in z['paths']))
result=dict(status='candidate_bounded_visual_sidecar',inspected_unique_rows=len(rows),physical_HOLD_rows=sum(x['physical']['classification']=='physical_identity_HOLD' for x in rows.values()),previous_history_HOLD_rows=[3,4],review='Full-image context for three branch families and all fourteen extents in local contact sheet; no new annotation or census.',annotation_count=len(bank),scorer=str(src),rows=rows,limitation='Single-row frozen class-agnostic IoU50 matches are not physical truth; unmatched UNKNOWN. HOLD rows cannot reverse absence of a retained nonrepeated row above native greedy because every branch30 row scores lower.')
(R/'physical-score.json').write_text(json.dumps(result,indent=2)+'\n');print([(k,v['scoring']['matches']['covered_owner_ids'],v['physical']['classification']) for k,v in rows.items()])
