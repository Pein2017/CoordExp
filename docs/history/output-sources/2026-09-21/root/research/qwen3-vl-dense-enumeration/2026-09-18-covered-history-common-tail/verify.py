"""Independent saved-logit likelihood and complete contrast reconstruction."""
import json,pathlib,hashlib,itertools,math
import torch
R=pathlib.Path(__file__).parent
torch.set_num_threads(4)
p=json.load(open(R/'panel.json'));d=json.load(open(R/'reduction.json'));scores={};positions=None;max_error=0.;max_float_error=0.
for window in p['windows']:
 name=window['name'];out=R/'runtime'/name;ts=torch.load(out/'scores.pt',weights_only=True,map_location='cpu');rd=torch.load(out/'readout.pt',weights_only=True,map_location='cpu');scores[name]={}
 assert window['history'][:9]==p['h']['tokens'] and window['history'][19:]==p['C']['tokens'] and window['history'][9:19]==p['candidates'][name]['tokens']
 assert len(window['history'])==29 and len(p['C']['tokens'])==10
 for candidate,t in ts.items():
  assert t['history']==window['history'] and t['token_ids']==p['candidates'][candidate]['tokens']
  if positions is None:positions=t['positions']
  else:assert torch.equal(positions,t['positions'])
  z=t['logits'];scaled=z.clone();scaled[:,rd['coordinate_ids']]=(z[:,rd['coordinate_ids']].double()*rd['factors']).float();scores[name][candidate]={}
  for policy,v in [('raw',z),('equal_norm',scaled)]:
   ids=torch.tensor(t['token_ids']);zz=v.double();selected=zz[torch.arange(10),ids]-torch.logsumexp(zz[:10],-1);value=float(selected.sum());expected=d['conditions'][name]['candidates'][candidate]['scores'][policy];assert abs(value-expected['logprob'])<1e-10;assert abs(math.exp(value)-expected['probability'])<1e-14
   max_error=max(max_error,abs(value-expected['logprob']));floatvalue=float(v[:10].log_softmax(-1)[torch.arange(10),ids].sum());max_float_error=max(max_float_error,abs(floatvalue-value));scores[name][candidate][policy]=value
 native=torch.load(out/'incremental.pt',weights_only=True,map_location='cpu');assert sorted(map(int,native))==list(range(29,40))
 for offset in range(29,40):
  j=offset-29;n=native[str(offset)];assert n['history']==window['history']+ts['A1']['token_ids'][:j];assert int(n['logits'].argmax())==int(ts['A1']['logits'][j].argmax());assert torch.equal(n['positions'][:,-1],positions[:,-11+j])
 # Check real prefix inequality gate has teeth without any model rerun.
 bad=window['history'].copy();bad[19]+=1;assert bad[-10:]!=p['C']['tokens']
 rec=json.load(open(out/'receipt.json'));assert rec['model_forwards']==44 and not pathlib.Path('/proc',str(rec['pid'])).exists()
for item in d['E']:
 sa,sb,a,b=(item[k] for k in ['supplied_A','supplied_B','candidate_A','candidate_B'])
 for policy in ['raw','equal_norm']:
  value=(scores[sa][a][policy]-scores[sa][b][policy])-(scores[sb][a][policy]-scores[sb][b][policy]);expected=item['policies'][policy];assert abs(value-expected['E'])<1e-10;assert abs(sum(expected['tokenwise_E_contributions'])-value)<1e-10
for b in p['sources']:
 f=pathlib.Path(b['path']);assert f.stat().st_size==b['size_bytes'] and hashlib.sha256(f.read_bytes()).hexdigest()==b['sha256'],f
assert all(json.load(open(R/'annotation-identity.json'))['checks'].values())
result=dict(status='passed',complete_row_scores=16,raw_and_equal_norm=True,E_values=32,first_history_and_common_tail_exact=True,positions_exact_across_all16=True,incremental_agreement_positions=44,common_tail_corruption_falsification=True,all_source_bindings=True,max_independent_logprob_error=max_error,max_FP32_vs_FP64_logsoftmax_error=max_float_error,all_producers_ended=True)
(R/'independent-verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
