import json,hashlib,math,collections
from pathlib import Path
R=Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-16-source256-output-ranking-repair');M=R/'runtime/main-v1'
read=lambda p:json.loads(p.read_text())
def bind(p):return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'size_bytes':p.stat().st_size}
verified={}
def verify(v):
 if isinstance(v,dict):
  if 'path' in v and 'sha256' in v:
   p=Path(v['path']); b=bind(p); assert b['sha256']==v['sha256'],str(p)
   if 'size_bytes' in v:assert b['size_bytes']==v['size_bytes'],str(p)
   verified[str(p)]=b
  for x in v.values():verify(x)
 elif isinstance(v,list):
  for x in v:verify(x)
d=read(M/'result.json');data=read(R/'preparation/prepared.json');ref=read(R/'preparation/reference-scores.json');refs={x['key']:x for x in ref['routes']}
verify(read(R/'lead/qualification-acceptance.json'));verify(d);verify(ref)
summary={'schema':'source256.ranking.lead_acceptance.v1','technical_status':'lead-accepted','scientific_status':'bounded_repair_not_promoted','bindings':{},'runtime':{},'likelihood':{},'metrics':{},'slices':{},'identity_turnover':d['actual91_starting_source_new'],'gates':d['gates']}
selected=set(d['selection']['image_ids'])
def compact(v):return {'primary':v['primary_class_agnostic_iou50'],'burden':v['burden']}
for label,s in d['scores'].items():
 summary['metrics'][label]={sp:compact(v) for sp,v in s['splits'].items()}
 rows=s['splits']['train']['per_image']; by={int(x['image_id']):x for x in rows}
 groups={'selected15':selected,'selected14':selected-{548337},'image548337':{548337},'outside241':set(by)-selected}
 summary['slices'][label]={}
 for name,ids in groups.items():
  matched=sum(by[i]['primary_class_agnostic_iou50']['matched_count'] for i in ids)
  targets=sum(by[i]['primary_class_agnostic_iou50']['target_count'] for i in ids)
  summary['slices'][label][name]={'matched':matched,'FN':targets-matched,'targets':targets}
for arm in ['P','R']:
 t=read(M/arm/'training/terminal.json');man=read(R/f'preparation/{arm}-main.json');lik=read(M/arm/'likelihood.json')
 verify(t);verify(man);verify(lik)
 assert t['status']=='completed' and t['updates']==16 and t['optimizer_mode']=='fresh'
 assert t['model_calls']==(512 if arm=='P' else 768)
 assert t['logical_model_forwards']==(1024 if arm=='P' else 1536)
 assert lik['checkpoint']['root']==str(M/arm/'training/checkpoints/step-00016')
 assert lik['manifest']==bind(R/f'preparation/{arm}-main.json') and lik['data']==ref['data']
 assert len(lik['routes'])==30 and lik['counts']['model_calls']==16
 canon=collections.Counter();pair=collections.Counter()
 for step in range(1,17):
  u=read(M/arm/f'training/updates/step-{step:05d}.json');schedule=data['schedule'][step-1]
  assert u['status']=='completed' and u['step']==step and math.isfinite(u['gradient_norm_before_clip'])
  for branch,expected,counter in [('canonical',schedule['common_image_ids'],canon),('pair',schedule['pair_image_ids'],pair)]:
   actual=[x['image_id'] for x in u['presentations'] if x['branch']==branch]
   assert collections.Counter(actual)==collections.Counter(expected),(arm,step,branch)
   counter.update(actual)
  if arm=='R':
   ranks=[]
   for row in u['presentations']:
    if row['branch']!='pair':continue
    rank=row['ranking'];i=str(row['image_id']);den=data['pairs'][i]['denominator']
    assert rank['recorded_length_denominator']==den and rank['lambda_value']==1
    assert row['rejected']['geometry_included'] is False
    if i=='548337':assert row['rejected']['active_tokens']==3084 and row['rejected']['eos_present'] is False
    z=-((rank['current_preferred_logp_sum']-rank['current_rejected_logp_sum'])-(rank['reference_preferred_logp_sum']-rank['reference_rejected_logp_sum']))/den
    expected=max(z,0)+math.log1p(math.exp(-abs(z)))
    assert math.isclose(rank['softplus_rank'],expected,abs_tol=2e-6)
    ranks.append(rank['softplus_rank'])
   assert math.isclose(sum(ranks)/32,u['ranking_mean'],abs_tol=1e-6)
  assert math.isclose(u['objective_total'],u['P_total']+(.5*u['ranking_mean'] if arm=='R' else 0),abs_tol=1e-6)
 assert sum(canon.values())==512 and set(canon.values())=={2}
 assert sum(pair.values())==512 and set(pair.values())=={34,35}
 summary['runtime'][arm]={'updates':16,'model_calls':t['model_calls'],'logical_model_forwards':t['logical_model_forwards'],'canonical_presentations':512,'pair_presentations':512,'pair_counts':dict(sorted(pair.items())),'adapter_fingerprint':t['checkpoints'][0]['adapter']['fingerprint'],'elapsed_seconds':t['elapsed_seconds']}
 routes={x['key']:x for x in lik['routes']};assert set(routes)==set(refs)
 per=[]
 for i in sorted(selected):
  plus=routes[f'{i}:preferred'];minus=routes[f'{i}:rejected'];den=data['pairs'][str(i)]['denominator']
  for v in [plus,minus]:
   old=refs[v['key']]
   for k in ['image_id','role','route_id','length','token_ids_sha256']:assert v[k]==old[k]
   assert math.isfinite(v['logp_sum'])
  dp=plus['logp_sum']-refs[plus['key']]['logp_sum'];dm=minus['logp_sum']-refs[minus['key']]['logp_sum']
  per.append({'image_id':i,'denominator':den,'preferred_delta_sum':dp,'rejected_delta_sum':dm,'preferred_delta_per_observed_token':dp/plus['length'],'rejected_delta_per_observed_token':dm/minus['length'],'anchored_pair_margin_delta':(dp-dm)/den})
 summary['likelihood'][arm]={'per_pair':per,'aggregates':{}}
 for name,ids in [('full15',selected),('descriptive14',selected-{548337}),('image548337',{548337})]:
  subset=[x for x in per if x['image_id'] in ids]
  summary['likelihood'][arm]['aggregates'][name]={k:sum(x[k] for x in subset)/len(subset) for k in per[0] if k not in ['image_id','denominator']}
for label in ['P16','R16']:
 source={(x['image_id'],str(o)) for x in d['scores']['Source0']['splits']['train']['per_image'] for o in x['primary_class_agnostic_iou50']['covered_owner_ids']}
 covered=lambda label:{(x['image_id'],str(o)) for x in d['scores'][label]['splits']['train']['per_image'] for o in x['primary_class_agnostic_iou50']['covered_owner_ids']}
 start=covered('Bnormalized64')-source;end=covered(label);s=d['actual91_starting_source_new'][label]
 assert len(start)==91 and len(start&end)==s['survived_count'] and len(start-end)==s['lost_count'] and len((end-source)-start)==s['replacement_gain_count']
assert read(R/'lead/recomputed-result.json')==d
verify(bind(M/'result.json'))
verify(bind(R/'lead/recomputed-result.json'))
verify(bind(Path(__file__)))
summary['recomputation']='Exact equality with independently rerun saved-row reducer; no model forwards.'
summary['bindings']=verified
(R/'lead/final-acceptance-v1.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
print(json.dumps({'verified_bindings':len(verified),'runtime':summary['runtime'],'slices':summary['slices'],'likelihood':{a:v['aggregates'] for a,v in summary['likelihood'].items()}},indent=2))
