"""Freeze eleven original recurrence boundaries and eight balanced sign vectors."""
import json,hashlib,random,datetime
from pathlib import Path
import torch
BASE=Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration');PRE=BASE/'2026-09-18-readout-common-component';OUT=BASE/'2026-09-19-readout-direction-control'
def bind(p):return dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def main():
 OUT.mkdir(exist_ok=True);assert not (OUT/'execution-plan.json').exists();old=json.loads((PRE/'execution-plan.json').read_text());selection=json.loads(Path(old['selection']).read_text());boundaries=[b for b in selection['boundaries'] if b['kind']=='failure'];assert len(boundaries)==11
 signs=[]
 for seed in range(19,27):
  plus=set(random.Random(seed).sample(range(1000),500));v=[1 if i in plus else -1 for i in range(1000)];payload=json.dumps(v,separators=(',',':')).encode();signs.append(dict(policy=f'sign{seed}',seed=seed,signs=v,sha256=hashlib.sha256(payload).hexdigest()))
 (OUT/'sign-vectors.json').write_text(json.dumps(dict(generator='Python random.Random(seed).sample(range(1000),500); selected=+1, remainder=-1',coordinate_ids=list(range(151670,152670)),vectors=signs),indent=2)+'\n')
 (OUT/'selection.json').write_text(json.dumps(dict(boundaries=boundaries,source=bind(Path(old['selection']))),indent=2)+'\n')
 policies=['original','full','reflected']+[f'sign{s}' for s in range(19,27)];cells=[]
 for b in boundaries:
  for policy in policies:
   c=dict(id=b['id']+'--'+policy,boundary_id=b['id'],model=b['model'],policy=policy,qualification=b['id']=='untied-885-failure')
   if policy in ['original','full']:
    ps=list((PRE/'runtime').glob('*/'+c['id']+'/release.json'));assert len(ps)==1;c['control_source']=bind(ps[0])
   cells.append(c)
 for model in ['tied','untied']:
  for i,c in enumerate(c for c in cells if c['model']==model and not c['qualification']):c['gpu']=i%4+(4 if model=='untied' else 0)
 for c in cells:
  if c['qualification']:c['gpu']=0
 feasibility=json.loads((BASE/'2026-09-18-history-readout-crossover/coordination/lead-direction-control-feasibility.json').read_text());ranges=[]
 for f in feasibility:
  p=Path(f['source']);assert bind(p)['sha256']==f['source_sha256'];w=torch.load(p,map_location='cpu',weights_only=False);W=w['output_rows'].double();norm=W.norm(dim=1);a=norm.median()/norm;assert bool(((2-a)>0).all());ranges.append(dict(model=f['model'],source=bind(p),alpha_min=float(a.min()),alpha_max=float(a.max()),reflected_min=float((2-a).min())))
 plan=dict(status='frozen_before_gpu',created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),selection=str(OUT/'selection.json'),panel=old['panel'],sign_vectors=bind(OUT/'sign-vectors.json'),bindings=[bind(OUT/'selection.json'),bind(Path(old['panel'])),bind(PRE/'lead-acceptance.json'),bind(PRE/'execution-plan.json')],policies=policies,cells=cells,coefficient_feasibility=ranges,bounds=dict(target_continuations=121,model_forwards=100000,gpu_seconds=28800,tensor_bytes=16*1024**3,target_tokens=512,target_rows=32),expected_max_release_forwards=121*512,qualification='All11 policies untied-885-failure reused in121; remaining110 on8 GPUs',arithmetic='full=z*alpha unchanged; reflected=z-(alpha-1)*z; sign=z+s*(alpha-1)*z; FP64 then originaldtype. Magnitude equality prior to finalFP32 rounding.',scope='Original11 component-study failure prefixes. No crossover4/8 histories; target-only outcomes.')
 (OUT/'execution-plan.json').write_text(json.dumps(plan,indent=2)+'\n');print(json.dumps(dict(cells=len(cells),signs=len(signs),ranges=ranges)))
if __name__=='__main__':main()
