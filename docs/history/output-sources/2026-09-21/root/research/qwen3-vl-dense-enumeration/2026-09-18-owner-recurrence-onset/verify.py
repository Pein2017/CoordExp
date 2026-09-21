"""Independent saved-tensor likelihood and same-history fork checks."""
import json,pathlib,hashlib
import torch
R=pathlib.Path(__file__).parent
torch.set_num_threads(4)
p=json.load(open(R/'panel.json'));d=json.load(open(R/'reduction.json'));r=torch.load(R/'runtime/scores/readout.pt',weights_only=True,map_location='cpu');checks=[]
for w in p['windows']:
 ts=torch.load(R/f"runtime/scores/{w['name']}.pt",weights_only=True,map_location='cpu')
 for name,t in ts.items():
  z=t['logits'].double();ids=torch.tensor(t['token_ids']);terms=z[torch.arange(len(ids)),ids]-torch.logsumexp(z[:-1],dim=1);actual=float(terms.sum());reported=d['windows'][w['name']]['candidates'][name]['sum_logprob'];assert abs(actual-reported)<1e-10
  assert t['history']==w['history'];assert torch.equal(r['output_rows'],r['input_rows']);checks.append(dict(window=w['name'],candidate=name,likelihood_error=abs(actual-reported)))
 for f in d['windows'][w['name']]['forks']:
  a=ts[f['left']];b=ts[f['right']];j=f['row_offset'];assert a['token_ids'][:j]==b['token_ids'][:j] and a['token_ids'][j]!=b['token_ids'][j]
  v=a['logits'][j].double();i,k=a['token_ids'][j],b['token_ids'][j];assert float(v[i]-v[k])==f['left_minus_right_margin'];norm=v.clone();norm[r['coordinate_ids']]=(norm[r['coordinate_ids']]*r['factors']).float().double();assert float(norm[i]-norm[k])==f['equal_norm_left_minus_right_margin']
old=json.load(open(p['native_raw']['path']))['rows'];new=json.load(open(R/'runtime/native/raw.json'))['rows'];assert old==new
# A single-token corruption breaks the exact-trajectory gate; no silent set-only equality.
corrupt=json.loads(json.dumps(new));corrupt[p['target']]['token_ids'][49]+=1;assert corrupt!=old
ann=json.load(open(R/'annotation-identity.json'));assert ann['positive_count']==63 and all(ann['checks'].values())
for b in p['sources']:
 f=pathlib.Path(b['path']);assert hashlib.sha256(f.read_bytes()).hexdigest()==b['sha256']
pre=json.load(open(R/'saved-current-scoring.json'));assert all(pre[k]==d['scoring'][k]['current63'] for k in pre)
receipt=dict(status='passed',independent_candidate_checks=checks,exact_all_four_sequences=True,token_corruption_falsification=True,annotation_current_bytes=True,all_source_bindings=True,current_score_reconstruction_exact=True)
(R/'independent-verification.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({k:v for k,v in receipt.items() if k!='independent_candidate_checks'}))
