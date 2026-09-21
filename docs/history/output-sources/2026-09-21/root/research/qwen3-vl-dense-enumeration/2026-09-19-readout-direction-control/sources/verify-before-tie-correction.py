"""Independent CPU operator/magnitude reconstruction and corruption sensitivity."""
import argparse,json,hashlib
from pathlib import Path
import torch

def verify(root):
 torch.set_num_threads(2);p=json.loads((root/'execution-plan.json').read_text());policies=p['policies'];vectors=json.loads(Path(p['sign_vectors']['path']).read_text())['vectors'];signs={v['policy']:torch.tensor(v['signs'],dtype=torch.float64) for v in vectors}
 for v in vectors:
  assert v['signs'].count(1)==v['signs'].count(-1)==500
  assert hashlib.sha256(json.dumps(v['signs'],separators=(',',':')).encode()).hexdigest()==v['sha256']
 cells={c['id']:c for c in p['cells']};reports=[];sensitivity=None
 for shard in sorted((root/'runtime').iterdir()):
  wp=shard/'effective-readout.pt'
  if not wp.exists():continue
  w=torch.load(wp,map_location='cpu',weights_only=False);W=w['output_rows'].double();n=W.norm(dim=1);a=n.median()/n;assert torch.allclose(a,w['factors'],atol=1e-12,rtol=0);assert bool(((2-a)>0).all())
  for rp in sorted(shard.glob('*/release.json')):
   r=json.loads(rp.read_text());t=torch.load(rp.parent/'trajectory.pt',map_location='cpu',weights_only=False);z=t['raw_coordinate_logits'].double();delta=(a-1)*z;fp64=[z,z*a,z-delta]+[z+signs[f'sign{s}']*delta for s in range(19,27)];expected=torch.stack(fp64,1).float();saved=t['operator_coordinate_logits'];err=float((expected-saved).abs().max());assert err<=2e-4,(rp,err)
   magnitude=max(float(((x-z).abs()-delta.abs()).abs().max()) for x in fp64[1:]);assert magnitude<1e-12
   rounded=max(float(((x.float().double()-z).abs()-delta.abs()).abs().max()) for x in fp64[1:])
   tokens=r['target']['token_ids'];steps=r['trace']['steps'];assert len(tokens)==len(steps)==len(t['head_inputs']);assert t['target_tokens'].tolist()==tokens
   for i,s in enumerate(steps):
    assert s['offset']==i and s['chosen_token']==s['emitted_token']==tokens[i]
    assert s['operators'][r['policy']]['top2'][0]['token_id']==tokens[i]
    for j,policy in enumerate(policies):
     winner=s['operators'][policy]['top2'][0]['token_id']
     if 151670<=winner<152670:assert int(saved[i,j].argmax())==winner-151670
   q=r.get('qualification')
   if q:
    assert q['identity_operator_bitwise'] and all(q['noncoordinate_bitwise_unchanged'].values())
    assert q['full_formula_max_abs_error']==0 and q['direction_magnitude_max_abs_error_fp64']<1e-12
   control=cells[r['job_id']].get('control_source');equal=None
   if control:
    cp=Path(control['path']);assert hashlib.sha256(cp.read_bytes()).hexdigest()==control['sha256'];equal=tokens==json.loads(cp.read_text())['target']['token_ids'];assert equal,r['job_id']
   if sensitivity is None:
    corrupt=saved.clone();corrupt[0,2,0]+=0.01;bad=float((corrupt-expected).abs().max());assert bad>2e-4;sensitivity=dict(original_error=err,corruption_error=bad,threshold=2e-4,detected=True,method='In-memory+0.01 corruption to one reflected score; saved artifacts unchanged.')
   reports.append(dict(cell=r['job_id'],steps=len(tokens),formula_max_abs=err,fp64_magnitude_max_abs=magnitude,fp32_change_magnitude_rounding_max_abs=rounded,unchanged_control_equal=equal,trajectory_sha256=hashlib.sha256((rp.parent/'trajectory.pt').read_bytes()).hexdigest()))
 return dict(status='candidate_CPU_verified',cells=len(reports),steps=sum(x['steps'] for x in reports),sensitivity=sensitivity,reports=reports)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();d=verify(a.root);a.out.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(dict(cells=d['cells'],steps=d['steps'])))
