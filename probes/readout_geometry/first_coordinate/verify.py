"""CPU audit of the sole token override, subsequent greedy choices and readout."""
import argparse,json,hashlib,copy
from pathlib import Path
import torch
PRELUDE=[151646,15007,332,151647,151648]
def intervention_check(tokens,steps,policy,forced):
 assert tokens[:5]==PRELUDE and tokens[5]==forced
 assert len(tokens)==len(steps) and len(steps)>6
 assert [i for i,s in enumerate(steps) if s['carrier_override_applied']]==[5]
 for i,s in enumerate(steps):
  assert s['offset']==i and s['emitted_token']==tokens[i] and s['carrier_winner_token']==tokens[i]
  if i:assert s['input_token_id']==tokens[i-1]
  if i==5:assert s['override_offset']==5 and s['forced_token']==forced
  else:assert s['operator_winner_token']==tokens[i] and s['override_offset'] is None
  assert s['operator_winner_token']==s['operators'][policy]['operator_winner_token']
 assert steps[6]['input_token_id']==forced

def verify(root):
 torch.set_num_threads(2);p=json.loads((root/'execution-plan.json').read_text());policies=p['policies'];vectors=json.loads(Path(p['sign_vectors']['path']).read_text())['vectors'];signs={v['policy']:torch.tensor(v['signs'],dtype=torch.float64) for v in vectors};cells={c['id']:c for c in p['cells']};reports=[];sensitivity=None
 for v in vectors:assert hashlib.sha256(json.dumps(v['signs'],separators=(',',':')).encode()).hexdigest()==v['sha256']
 for shard in sorted((root/'runtime').iterdir()):
  wp=shard/'effective-readout.pt'
  if not wp.exists():continue
  w=torch.load(wp,map_location='cpu',weights_only=False);W=w['output_rows'].double();n=W.norm(dim=1);a=n.median()/n;assert torch.allclose(a,w['factors'],atol=1e-12,rtol=0)
  for rp in sorted(shard.glob('*/release.json')):
   r=json.loads(rp.read_text());c=cells[r['job_id']];t=torch.load(rp.parent/'trajectory.pt',map_location='cpu',weights_only=False);z=t['raw_coordinate_logits'].double();delta=(a-1)*z;expected=torch.stack([z]+[z+signs[pol]*delta for pol in policies[1:]],1).float();saved=t['operator_coordinate_logits'];err=float((expected-saved).abs().max());assert err<=2e-4
   tokens=r['target']['token_ids'];steps=r['trace']['steps'];intervention_check(tokens,steps,c['policy'],c['forced_token']);assert t['target_tokens'].tolist()==tokens;assert len(t['head_inputs'])==len(tokens)
   idx=policies.index(c['policy']);applied=saved[:,idx];assert torch.equal(applied,t['applied_coordinate_logits']);carrier=t['carrier_coordinate_logits'];mask=torch.ones(len(tokens),dtype=torch.bool);mask[5]=False;assert torch.equal(carrier[mask],applied[mask]);forced_vector=torch.full_like(carrier[5],torch.finfo(carrier.dtype).min);forced_vector[c['forced_choice']]=0.;assert torch.equal(carrier[5],forced_vector);assert t['override_flags'].nonzero().flatten().tolist()==[5]
   for i,s in enumerate(steps):
    for j,pol in enumerate(policies):
     winner=s['operators'][pol]['operator_winner_token']
     if 151670<=winner<152670:assert saved[i,j,winner-151670]==saved[i,j].max()
   equal=None
   if c['concordant']:
    cp=Path(c['control_source']['path']);assert hashlib.sha256(cp.read_bytes()).hexdigest()==c['control_source']['sha256'];equal=tokens==json.loads(cp.read_text())['target']['token_ids'];assert equal
   if sensitivity is None:
    detected=[]
    for kind in ['wrong_offset','wrong_token']:
     bt=tokens.copy();st=copy.deepcopy(steps)
     if kind=='wrong_offset':st[5]['override_offset']=6
     else:bt[5]=151670+(1-c['forced_choice'])
     try:intervention_check(bt,st,c['policy'],c['forced_token'])
     except AssertionError:detected.append(kind)
    assert len(detected)==2;sensitivity=dict(detected=detected,method='Corrupt CPU copies of actual saved offset/token; original artifacts unchanged.')
   reports.append(dict(cell=r['job_id'],steps=len(tokens),formula_max_abs=err,override_offset=5,forced_token=c['forced_token'],operator_winner_before_override=steps[5]['operator_winner_token'],concordant=c['concordant'],control_equal=equal,trajectory_sha256=hashlib.sha256((rp.parent/'trajectory.pt').read_bytes()).hexdigest()))
 return dict(status='candidate_CPU_verified',cells=len(reports),steps=sum(x['steps'] for x in reports),sensitivity=sensitivity,reports=reports)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();d=verify(a.root);a.out.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(dict(cells=d['cells'],steps=d['steps'])))
