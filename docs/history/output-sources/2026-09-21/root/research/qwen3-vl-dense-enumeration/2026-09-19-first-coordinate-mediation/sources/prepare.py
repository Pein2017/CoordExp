"""Freeze reciprocal first-x1 intervention; selection is explicitly post-hoc."""
import json,hashlib,datetime
from pathlib import Path
BASE=Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration');PRE=BASE/'2026-09-19-readout-direction-control';OUT=BASE/'2026-09-19-first-coordinate-mediation'
def bind(p):return dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def main():
 OUT.mkdir(exist_ok=True);assert not (OUT/'execution-plan.json').exists();old=json.loads((PRE/'execution-plan.json').read_text());selection=json.loads(Path(old['selection']).read_text());bs=[b for b in selection['boundaries'] if b['image_id']==417044];assert len(bs)==2
 (OUT/'selection.json').write_text(json.dumps(dict(boundaries=bs,source=bind(Path(old['selection']))),indent=2)+'\n');policies=['original']+[f'sign{s}' for s in range(19,27)];cells=[];controls=[]
 for b in bs:
  for policy in policies:
   p=list((PRE/'runtime').glob('*/'+b['id']+'--'+policy+'/release.json'));assert len(p)==1;p=p[0];r=json.loads(p.read_text());tokens=r['target']['token_ids'];assert tokens[:5]==[151646,15007,332,151647,151648];native=tokens[5]-151670;assert native==int(policy in ['sign20','sign22','sign23','sign24']);controls.append(dict(boundary_id=b['id'],policy=policy,native_choice=native,source=bind(p)))
   for forced in [0,1]:
    c=dict(id=b['id']+'--'+policy+f'--x1-{forced}',boundary_id=b['id'],model=b['model'],policy=policy,forced_choice=forced,forced_offset=5,forced_token=151670+forced,native_choice=native,concordant=forced==native,qualification=b['model']=='untied' and policy in ['original','sign20'],source_policy=bind(p))
    if c['concordant']:c['control_source']=bind(p)
    cells.append(c)
 for model in ['tied','untied']:
  for i,c in enumerate(c for c in cells if c['model']==model and not c['qualification']):c['gpu']=i%4+(4 if model=='untied' else 0)
 for c in cells:
  if c['qualification']:c['gpu']=0
 plan=dict(status='frozen_before_gpu',created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),selection=str(OUT/'selection.json'),panel=old['panel'],sign_vectors=old['sign_vectors'],policies=policies,cells=cells,controls=controls,expected_prelude=[151646,15007,332,151647,151648],bindings=[bind(OUT/'selection.json'),bind(Path(old['panel'])),bind(PRE/'lead-acceptance.json'),bind(PRE/'execution-plan.json'),bind(PRE/'coordination/lead-donut-first-coordinate-admission.json')],bounds=dict(target_continuations=36,model_forwards=50000,gpu_seconds=14400,tensor_bytes=8*1024**3,release_tokens=512,release_rows=32),expected_max_release_forwards=36*512,scope='Outcome-selected donut follow-up; only offset5 emitted choice overridden, allother target tokens nativegreedy underdeclaredreadout. Primary excludes first completed partiallycontrolledrow; whole metrics separate.')
 (OUT/'execution-plan.json').write_text(json.dumps(plan,indent=2)+'\n');print(json.dumps(dict(cells=len(cells),concordant_controls=sum(c['concordant'] for c in cells),qualification=sum(c['qualification'] for c in cells))))
if __name__=='__main__':main()
