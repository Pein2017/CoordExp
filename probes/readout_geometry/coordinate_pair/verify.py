"""CPU support arithmetic, untouched-score identity and wrong-support detection."""
import argparse,json,hashlib
from pathlib import Path
import torch

def verify(root):
 torch.set_num_threads(2);p=json.loads((root/'execution-plan.json').read_text());policies=p['policies'];vec=next(v for v in json.loads(Path(p['sign_vectors']['path']).read_text())['vectors'] if v['policy']=='sign23');assert hashlib.sha256(json.dumps(vec['signs'],separators=(',',':')).encode()).hexdigest()==vec['sha256'];sign=torch.tensor(vec['signs'],dtype=torch.float64);cells={c['id']:c for c in p['cells']};reports=[];sensitivity=None
 for wp in sorted((root/'runtime').rglob('effective-readout.pt')):
  shard=wp.parent
  w=torch.load(wp,map_location='cpu',weights_only=False);W=w['output_rows'].double();n=W.norm(dim=1);a=n.median()/n;assert torch.allclose(a,w['factors'],atol=1e-12,rtol=0)
  for rp in sorted(shard.glob('*/release.json')):
   r=json.loads(rp.read_text());t=torch.load(rp.parent/'trajectory.pt',map_location='cpu',weights_only=False);z=t['raw_coordinate_logits'].double();delta=sign*(a-1)*z;expected=[];increments={}
   for policy in policies:
    support=p['supports'][policy];x=z.clone();x[:,support]+=delta[:,support];expected.append(x.float());increments[policy]=x-z
   expected=torch.stack(expected,1);saved=t['operator_coordinate_logits'];err=float((expected-saved).abs().max());assert err<=2e-4
   decomposition=max(float((increments['only0']+increments['only1']-increments['pair01']).abs().max()),float((increments['pair01']+increments['except01']-increments['sign23']).abs().max()));assert decomposition<1e-12
   for j,policy in enumerate(policies):
    outside=torch.ones(1000,dtype=torch.bool);outside[p['supports'][policy]]=False;assert torch.equal(saved[:,j,outside].contiguous().view(torch.int32),t['raw_coordinate_logits'][:,outside].contiguous().view(torch.int32))
   tokens=r['target']['token_ids'];steps=r['trace']['steps'];assert len(tokens)==len(steps)==len(t['head_inputs']);assert t['target_tokens'].tolist()==tokens
   for i,s in enumerate(steps):
    assert s['offset']==i and s['chosen_token']==s['emitted_token']==tokens[i]
    chosen=s['operators'][r['policy']].get('operator_winner_token',s['operators'][r['policy']]['chosen_token']);assert chosen==tokens[i]
    for j,pol in enumerate(policies):
     winner=s['operators'][pol]['top2'][0]['token_id']
     if 151670<=winner<152670:assert saved[i,j,winner-151670]==saved[i,j].max()
   q=r.get('qualification')
   if q:assert all(q['noncoordinate_bitwise_unchanged'].values())
   c=cells[r['job_id']];equal=None
   if 'control_source' in c:
    cp=Path(c['control_source']['path']);assert hashlib.sha256(cp.read_bytes()).hexdigest()==c['control_source']['sha256'];equal=tokens==json.loads(cp.read_text())['target']['token_ids'];assert equal
   if sensitivity is None:
    bad=saved.clone();bad[0,policies.index('only0'),2]+=0.01;outside=torch.ones(1000,dtype=torch.bool);outside[0]=False;detected=not torch.equal(bad[:,policies.index('only0'),outside],t['raw_coordinate_logits'][:,outside]);assert detected;sensitivity=dict(wrong_support_bin=2,policy='only0',detected=True,method='Add0.01 on an unsupported bin in CPU copy; exact untouched-score check rejects it.')
   reports.append(dict(cell=r['job_id'],steps=len(tokens),formula_max_abs=err,decomposition_max_abs=decomposition,control_equal=equal,trajectory_sha256=hashlib.sha256((rp.parent/'trajectory.pt').read_bytes()).hexdigest()))
 assert reports, 'No cell tensors discovered'
 return dict(status='candidate_CPU_verified',cells=len(reports),steps=sum(x['steps'] for x in reports),sensitivity=sensitivity,reports=reports)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();d=verify(a.root);a.out.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(dict(cells=d['cells'],steps=d['steps'])))
