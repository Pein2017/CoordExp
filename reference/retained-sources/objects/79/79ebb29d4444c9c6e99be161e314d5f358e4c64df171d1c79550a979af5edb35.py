import json,re,hashlib
from pathlib import Path
from transformers import AutoTokenizer
R=Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-16-source256-output-ranking-repair');O=R.parent/'2026-09-16-corner-loop-mechanism'
read=lambda p:json.loads(p.read_text())
def bind(p):return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def digest(v):return hashlib.sha256(json.dumps(v,separators=(',',':')).encode()).hexdigest()
m=read(R/'preparation/P-main.json');data=read(R/'preparation/prepared.json');tok=AutoTokenizer.from_pretrained(m['model_config']['model']['base_model'],local_files_only=True)
pattern=re.compile(r'<\|object_ref_start\|>(.*?)<\|object_ref_end\|><\|box_start\|>'+r'<\|coord_(\d+)\|>'*4+r'<\|box_end\|>')
def rows(ids):
 text=tok.decode(ids,skip_special_tokens=False);out=[]
 for match in pattern.finditer(text):
  prefix=tok.encode(text[:match.start()],add_special_tokens=False);t=tok.encode(match.group(),add_special_tokens=False)
  assert ids[:len(prefix)]==prefix and ids[len(prefix):len(prefix)+len(t)]==t
  out.append({'start':len(prefix),'tokens':t,'text':match.group(),'category':match[1],'box':list(map(int,match.groups()[1:]))})
 return out
panel={'schema':'corner-loop.phase1.v1','status':'frozen','config':m['model_config'],'sources':[bind(R/'preparation/P-main.json'),bind(R/'preparation/prepared.json'),bind(R/'lead/final-acceptance-v1.json')],'checkpoints':{'Bnormalized64':m['model_config']['adapter']['path'],'P16':str(R/'runtime/main-v1/P/training/checkpoints/step-00016/adapter'),'R16':str(R/'runtime/main-v1/R/training/checkpoints/step-00016/adapter')},'cases':[],'boundary_rows':{'477415':[4,16,138],'351017':[1,2,16],'417044':[1,2,16]},'parity_rule':'Report full-vocabulary max discrepancy epsilon and relevant margin; material if stable argmax flips with winning gap > 1e-3 and >10x repeat-full no-op epsilon. Tiny flips within 1e-3 flagged numerical ambiguity, not bug. Any position/media identity mismatch stops attribution.','limits':{'cells':9,'boundaries':27,'candidate_rows_per_boundary':2,'model_forwards_max':6500,'new_generated_tokens':0,'wall_seconds_per_cell':1800},'batch_shape':'primary cache/full bs1 identical; duplicate-image/history bs4 full-prefix comparison at every boundary, not recreation of original heterogeneous natural bs4 batch'}
for iid in [477415,351017,417044]:
 route=data['canonical_routes'][str(iid)];raw={a:read(R/f'visuals/human-review-pr16-v1/{iid}-{a}-source.json') for a in ['P','R']};rr={a:rows(v['saved_generation']['generated_token_ids']) for a,v in raw.items()};panel['sources'] += [bind(R/f'visuals/human-review-pr16-v1/{iid}-{a}-source.json') for a in ['P','R']]
 p=raw['P']['saved_generation'];assert p['prompt_token_ids']==route['prompt_token_ids']
 diff=next(j+1 for j,(x,y) in enumerate(zip(rr['P'],rr['R'])) if x['tokens']!=y['tokens']);assert diff=={477415:4,351017:2,417044:2}[iid]
 census={a:{'complete_rows':len(xs),'x1_zero':sum(x['box'][0]==0 for x in xs),'geometry_invalid':sum(not(x['box'][0]<x['box'][2] and x['box'][1]<x['box'][3]) for x in xs),'corner_0_999_999_999':sum(x['box']==[0,999,999,999] for x in xs)} for a,xs in rr.items()}
 canonical=rows(route['continuation_token_ids']);owners=route['provenance']['suffix_owner_ids'];assert len(canonical)==len(owners)
 case={'image_id':iid,'case':route['case'],'image_identity':route['image_identity'],'prompt_ids':p['prompt_token_ids'],'P_action_ids':p['generated_token_ids'],'P_saved_generation_sha256':p['generated_token_ids_sha256'],'census':census,'first_PR_differing_row':diff,'first_PR_rows':{a:rr[a][diff-1] for a in ['P','R']},'boundaries':[]}
 def iou(a,b):
  inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]));aa=max(0,a[2]-a[0])*max(0,a[3]-a[1]);bb=(b[2]-b[0])*(b[3]-b[1]);return inter/(aa+bb-inter) if aa+bb-inter else 0
 for n in panel['boundary_rows'][str(iid)]:
  obs=rr['P'][n-1];prior=rr['P'][:n-1];alt=next((j,x) for j,x in enumerate(canonical) if x['box'][0]>0 and all(iou(q['box'],x['box'])<.5 for q in prior))
  case['boundaries'].append({'row_1based':n,'offset':obs['start'],'history_sha256':digest(p['generated_token_ids'][:obs['start']]),'observed':obs,'alternative':{**alt[1],'owner_id':owners[alt[0]],'provenance':'complete canonical trusted-bank owner row; not uniquely legal ordering; not matched by supplied prefix under pairwise IoU50'},'conditioning':'literal P history; off-policy for Bnormalized64/R16 after divergence; candidate row internal prefixes are teacher-forced scoring only'})
 panel['cases'].append(case)
 print(iid,census,'firstdiff',diff,'alternatives',[(b['row_1based'],b['alternative']['owner_id'],b['alternative']['box']) for b in case['boundaries']])
for name,path in panel['checkpoints'].items():
 panel['sources'] +=[bind(p) for p in sorted(Path(path).iterdir()) if p.is_file()]
emb=Path(m['model_config']['embedding_delta']['path']);panel['sources'] +=[bind(p) for p in sorted(emb.iterdir()) if p.is_file()]
(O/'panel.json').write_text(json.dumps(panel,indent=2)+'\n')
