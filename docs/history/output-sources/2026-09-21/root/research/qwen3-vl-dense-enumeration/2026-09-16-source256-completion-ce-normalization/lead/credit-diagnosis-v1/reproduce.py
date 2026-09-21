import json, hashlib
from pathlib import Path
from collections import Counter
from src.inference.parsing import parse_compact_object_box_closed
B=Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration')
O=B/'2026-09-16-source256-fixed-prefix-completion';N=B/'2026-09-16-source256-completion-ce-normalization'
def read(p):return json.loads(Path(p).read_text())
def bind(p):p=Path(p);return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
o=read(O/'runtime/main-v1/evaluation/result.json');n=read(N/'runtime/main-normalized-v1/evaluation/result.json');p=read(O/'preparation/source256-admitted-v1/preparation.json')
assert bind(O/'runtime/main-v1/evaluation/result.json')['sha256']=='1c45305af5c3b7689e75465092773437005a312632946ad2a83e706e48a8083d'
assert bind(N/'runtime/main-normalized-v1/evaluation/result.json')['sha256']=='d4674776e5ae509923462148028f428dda3b28c12af61789c82919827fb2cfa4'
assert bind(O/'preparation/source256-admitted-v1/preparation.json')['sha256']=='9b94baeb0699e479413483cba5ee6fb4b4d98c062aebb0ba462ae61a871cf242'
scores={**o['scores'],**n['new_scores']}
def loss(label):
 d=n if label.startswith('Bnormalized') else o
 return {(x['image_id'],x['owner_id']) for x in d['comparisons'][label+'_vs_Source0']['splits']['train']['primary_class_agnostic_iou50']['lost']}
a,z=loss('A64'),loss('Bnormalized64');pop=sorted(z-a);assert len(pop)==28 and len({x[0] for x in pop})==18
bank={x['image_id']:x['owners'] for x in p['bank']['records']};routes={x['image_id']:x['eligibility'] for x in p['routes']};dims={x['image_id']:x for x in map(json.loads,Path(p['sources']['train_jsonl']['path']).read_text().splitlines())}
def iou(a,b):
 area=lambda c:max(0,c[2]-c[0])*max(0,c[3]-c[1]);s=[max(a[0],b[0]),max(a[1],b[1]),min(a[2],b[2]),min(a[3],b[3])];i=area(s);return i/(area(a)+area(b)-i) if area(a)+area(b)-i else 0
images={};bindings=[bind(O/'runtime/main-v1/evaluation/result.json'),bind(N/'runtime/main-normalized-v1/evaluation/result.json'),bind(O/'preparation/source256-admitted-v1/preparation.json'),bind(p['sources']['train_jsonl']['path'])]
for label in ['Source0','A64','B64','Bnormalized64']:
 score={x['image_id']:x for x in scores[label]['splits']['train']['per_image']}
 for sb in scores[label]['readback_shards']['train']:
  s=read(sb['path']);assert bind(sb['path'])['sha256']==sb['sha256'];bindings.append(bind(sb['path']))
  for j,r in enumerate(s['generation']['rows']):
   im=r['image_id']
   if im not in {x[0] for x in pop}:continue
   assert r['actual_batch_size']==4
   v=parse_compact_object_box_closed(r['raw_decode_text'],row_id=r['example_id'],row_index=r['row_index'],image_width=dims[im]['width'],image_height=dims[im]['height']).to_artifact_dict()
   sc=score[im];mt={m['prediction_order']:m for m in sc['primary_class_agnostic_iou50']['matches']}
   preds=[dict(order=x['generated_order'],desc=x['description'],box=x['coord_bins'],span=[x['char_start'],x['char_end']],assigned_owner=mt.get(x['generated_order'],{}).get('reference_owner_id')) for x in v['predictions']]
   assert len(preds)==sc['valid_prediction_count']
   images.setdefault(im,{})[label]=dict(pointer=sb['path']+'#/generation/rows/'+str(j),token_sha=r['generated_token_ids_sha256'],prompt_sha=r['prompt_token_ids_sha256'],media_sha=r['executed_media_sha256'],predictions=preds,dropped=v['dropped_predictions'],burden=sc['burden'],stop=r['decode_stop_reason'],tokens=len(r['generated_token_ids']),matched=sc['primary_class_agnostic_iou50']['matched_count'],missing=sc['primary_class_agnostic_iou50']['missing_owner_ids'],matches=sc['primary_class_agnostic_iou50']['matches'])
rows=[]
for im,owner in pop:
 target=next(x for x in bank[im] if x['owner_id']==owner);el=routes[im];stratum='fallback' if not el['fully_eligible'] else 'prefix' if owner in el['prefix_owner_ids'] else 'suffix'
 v=dict(image_id=im,owner_id=owner,description=target['description'],target_box=target['coord_bins'],stratum=stratum,endpoints={})
 for label,d in images[im].items():
  near=sorted([dict(**x,iou=iou(target['coord_bins'],x['box'])) for x in d['predictions']],key=lambda x:x['iou'],reverse=True)[:3]
  v['endpoints'][label]=dict(nearest=near,owner_match=next((m for m in d['matches'] if m['reference_owner_id']==owner),None))
 assert len({d['prompt_sha'] for d in images[im].values()})==1 and len({d['media_sha'] for d in images[im].values()})==1
 rows.append(v)
strata={label:dict(Counter('fallback' if not routes[im]['fully_eligible'] else 'prefix' if owner in routes[im]['prefix_owner_ids'] else 'suffix' for im,owner in loss(label))) for label in ('A64','B64','Bnormalized64')}
out=dict(population=rows,images=images,bank={im:bank[im] for im in images},routes={im:routes[im] for im in images},bindings=bindings,strata=strata,set_counts=dict(primary=len(pop),reverse=len(a-z),common=len(a&z)))

# Explicit lead adjudications from retained row geometry; these are not new labels.
geo_images={25274,64010,101636,133279,143132,201145,234328}
hold={(102420,'1285990'),(527822,'1521959'),(158044,'1663172'),(158044,'1663276'),(203986,'source256-unlabeled-candidate-5937568de085ca903862'),(203986,'source256-unlabeled-candidate-5bda81ea3b2033331aa7')}
for v in rows:
 im,owner=v['image_id'],v['owner_id'];near=v['endpoints']['Bnormalized64']['nearest'][0]
 if im in geo_images or (im,owner)==(203986,'2029641'):
  category='1-localization-candidate';credit='geometry/extent correction candidate; physical identity not certified'
  reason='Source and A matched row persists as a nearby same-description B-normalized row with changed extent/location and IoU below .5; do not call it an absent physical object.'
 elif im==536467:
  category='1-assignment-competition';credit='HOLD local negative; preserve legal pizza, recover separate table obligation'
  reason='Pizza row IoU .595 with table is assigned to pizza. Reassigning it to table merely trades an owner, not a net repair; pizza is not a wrong local action.'
 elif (im,owner) in hold:
  category='4-HOLD';credit='HOLD owner-specific local action; route preference only when image dominance holds'
  reason='Crowded/overlapping or broad same-category rows leave physical identity or extent unresolved from saved text; no added witness.'
 elif im==548337:
  category='2-invalid-output-cap';credit='local grammar/valid-box constraint only; no earliest harmful token or recovered-owner guarantee'
  reason='One shared image-level failure: 291 parser geometry-invalid drops, one malformed span, 44 strict-repeat proxy rows and length cap. These six owner deficits are not six independent failures.'
 else:
  category='2-known-obligation-EOS';credit='limited EOS-too-early credit relative to frozen trusted bank; positive continuation unverified at this exact history'
  reason='Natural EOS with this trusted obligation uncovered and no defensible saved same-owner row; the complete Source/A routes cover it. This does not establish scene exhaustiveness.'
 v.update(category=category,credit=credit,reason=reason,secondary_tags=['route-level known-owner comparison; no causal token attribution'])
for im,ds in images.items():
 a,norm=ds['A64'],ds['Bnormalized64'];cover=lambda d:{m['reference_owner_id'] for m in d['matches']};debt=lambda d:[len(d['dropped']),len(d['burden']['strict_repeat_rows']),int(d['stop']=='length')]
 ds['route_comparison']=dict(A_only=sorted(cover(a)-cover(norm)),normalized_only=sorted(cover(norm)-cover(a)),A_debt=debt(a),normalized_debt=debt(norm),strict_known_owner_debt_dominance=cover(a)>cover(norm) and all(x<=y for x,y in zip(debt(a),debt(norm))),credit='complete-continuation preference under known-bank/proxy metrics only; no unknown-row, description or local-token correctness guarantee')
# Independent population cross-check from endpoint covered sets.
covered=lambda label:{(im['image_id'],oid) for im in scores[label]['splits']['train']['per_image'] for oid in im['primary_class_agnostic_iou50']['covered_owner_ids']}
assert set(pop)==(covered('Source0')&covered('A64'))-covered('Bnormalized64')
assert out['set_counts']==dict(primary=28,reverse=8,common=27)
assert strata=={'A64':{'prefix':11,'suffix':17,'fallback':7},'B64':{'prefix':20,'suffix':39,'fallback':16},'Bnormalized64':{'prefix':12,'suffix':31,'fallback':12}}
for im,ds in images.items():
 assert all(d['pointer'] for k,d in ds.items() if k!='route_comparison')
counts=dict(Counter(v['category'] for v in rows));assert sum(counts.values())==28
out.update(schema='source256.bounded_credit_diagnosis.v1',category_counts=counts,strict_route_preference_images=[im for im,ds in images.items() if ds['route_comparison']['strict_known_owner_debt_dominance']],limits=['posthoc selected 28 owners / 18 images, not population prevalence','No forwards, sampling, labels, images or new admission','Geometry/assignment matches do not certify physical identity','A saved A suffix is not a verified continuation at normalized history','No earliest harmful token or logit-margin claim','Unknown rows remain unknown; route preference is not correctness of every token','Raw parser geometry_invalid is included in legacy malformed counter; no retrospective metric redefinition'])
assert len(out['strict_route_preference_images'])==15
out['producer']=bind(__file__)
dest=N/'lead/credit-diagnosis-v1';dest.mkdir(exist_ok=True)
(dest/'diagnosis.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(counts=counts,strict_route_preference_images=len(out['strict_route_preference_images']),output=str(dest/'diagnosis.json'))))
