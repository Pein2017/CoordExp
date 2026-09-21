"""Reconstruct all conditional row, fork and covered-history contrasts on CPU."""
import argparse,json,pathlib,hashlib,itertools
import torch
R=pathlib.Path(__file__).parent

def bind(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),size_bytes=p.stat().st_size)
def read(p):return json.loads(pathlib.Path(p).read_text())
def rank(z,t):return 1+int((z>z[t]).sum())
def top(z):
 v,i=z.topk(2);return dict(token_id=int(i[0]),margin=float(v[0]-v[1]))
def reduce():
 p=read(R/'panel.json');result=dict(schema='covered_history.matrix.v1',conditions={},D={},E=[],fork_E=[],numerical=[],cost={},physical=read(R/'review/admission.json'))
 reference_positions=None;reference_rows=None
 for w in p['windows']:
  name=w['name'];out=R/'runtime'/name;rec=read(out/'receipt.json');assert rec['status']=='candidate_complete' and rec['model_forwards']==44 and rec['parameters_unchanged'];assert (R/f'{name}.exit').read_text().strip()=='0';assert rec['panel']==bind(R/'panel.json')
  ts=torch.load(out/'scores.pt',map_location='cpu',weights_only=True);rd=torch.load(out/'readout.pt',map_location='cpu',weights_only=True);native=torch.load(out/'incremental.pt',map_location='cpu',weights_only=True)
  W=rd['output_rows'].double();norm=rd['norms'];f=rd['factors'];coords=rd['coordinate_ids'];assert rd['bias'] is None and torch.equal(rd['input_rows'],rd['output_rows']);assert torch.equal(norm,W.norm(dim=1)) and torch.equal(f,norm.median()/norm)
  if reference_rows is None:reference_rows=W
  else:assert torch.equal(reference_rows,W)
  condition=dict(history_token_ids=w['history'],prefix_sha256=hashlib.sha256(json.dumps(w['history']).encode()).hexdigest(),supplied=w['supplied'],candidates={},forks=[])
  for candidate,t in ts.items():
   assert t['history']==w['history'] and t['token_ids']==p['candidates'][candidate]['tokens'];z=t['logits'].double();h=t['head_input'].double();ids=t['token_ids'];assert len(z)==len(h)==11
   if reference_positions is None:reference_positions=t['positions']
   else:assert torch.equal(reference_positions,t['positions']),'positions differ across matched histories/candidates'
   scaled=z.clone();scaled[:,coords]=(z[:,coords]*f).float().double();assert torch.equal(z[:,:151670],scaled[:,:151670]) and torch.equal(z[:,152670:],scaled[:,152670:])
   recon=h@W.T;err=float((recon-z[:,coords]).abs().max());assert err<1e-4
   slots=[];probabilities={};tokenprobs={}
   for policy,zz in [('raw',z),('equal_norm',scaled)]:
    lp=zz.log_softmax(-1);chosen=lp[torch.arange(10),torch.tensor(ids)];ll=float(chosen.sum());probabilities[policy]=dict(logprob=ll,probability=float(torch.exp(chosen.sum())));tokenprobs[policy]=chosen.tolist()
   for j in range(11):
    role='next_boundary' if j==10 else 'opener' if j==0 else 'description' if j in [1,2] else ['x1','y1','x2','y2'][j-5] if 5<=j<=8 else 'syntax'
    slots.append(dict(row_offset=j,action_offset=29+j,field=role,raw_top=top(z[j]),equal_norm_top=top(scaled[j]),raw_eos_logit=float(z[j,151645]),raw_eos_rank=rank(z[j],151645),opener_minus_eos=float(z[j,151646]-z[j,151645]),chosen_token=ids[j] if j<10 else None,raw_chosen_rank=rank(z[j],ids[j]) if j<10 else None,equal_norm_chosen_rank=rank(scaled[j],ids[j]) if j<10 else None,coordinate_argmax=int(z[j,coords].argmax()),equal_norm_coordinate_argmax=int(scaled[j,coords].argmax())))
   condition['candidates'][candidate]=dict(scores=probabilities,token_logprobs=tokenprobs,slots=slots,token_ids=ids,owner_id=p['candidates'][candidate]['owner_id'],readout_reconstruction_max_abs=err,tensor=bind(out/'scores.pt'))
  for a,b in itertools.combinations(ts,2):
   aa,bb=ts[a]['token_ids'],ts[b]['token_ids'];j=next(j for j in range(10) if aa[j]!=bb[j]);assert aa[:j]==bb[:j];v=ts[a]['logits'][j].double();other=ts[b]['logits'][j].double();eps=float((v-other).abs().max());scaled=v.clone();scaled[coords]=(v[coords]*f).float().double();marg=float(v[aa[j]]-v[bb[j]]);smarg=float(scaled[aa[j]]-scaled[bb[j]])
   condition['forks'].append(dict(left=a,right=b,row_offset=j,action_offset=29+j,left_token=aa[j],right_token=bb[j],raw_margin=marg,equal_norm_margin=smarg,raw_left_rank=rank(v,aa[j]),raw_right_rank=rank(v,bb[j]),equal_norm_left_rank=rank(scaled,aa[j]),equal_norm_right_rank=rank(scaled,bb[j]),raw_top=top(v),equal_norm_top=top(scaled),max_abs_same_prefix_suffix_error=eps,raw_pair_numerically_resolved=abs(marg)>2*eps,equal_norm_pair_numerically_resolved=abs(smarg)>2*eps))
  parity=[]
  for j in range(11):
   n=native[str(29+j)];a=ts['A1'];assert n['history']==w['history']+a['token_ids'][:j];v=n['logits'].double();z=a['logits'][j].double();eps=float((v-z).abs().max());margin=top(v)['margin'];assert int(v.argmax())==int(z.argmax()) and 2*eps<margin;assert torch.equal(a['positions'][:,-11+j],n['positions'][:,-1]);parity.append(dict(row_offset=j,max_abs=eps,twice_error_over_margin=2*eps/margin))
  condition['parity']=parity;result['conditions'][name]=condition;result['numerical'].append(dict(condition=name,max_readout_error=max(v['readout_reconstruction_max_abs'] for v in condition['candidates'].values()),max_cache_full_error=max(v['max_abs'] for v in parity),max_twice_error_over_margin=max(v['twice_error_over_margin'] for v in parity)))
  result['D'][name]={f'{a}-{b}':{policy:condition['candidates'][a]['scores'][policy]['logprob']-condition['candidates'][b]['scores'][policy]['logprob'] for policy in ['raw','equal_norm']} for a,b in itertools.product(['A1','A2'],['B1','B2'])}
 for sa,sb,a,b in itertools.product(['A1','A2'],['B1','B2'],['A1','A2'],['B1','B2']):
  entry=dict(supplied_A=sa,supplied_B=sb,candidate_A=a,candidate_B=b,policies={})
  fe=dict(supplied_A=sa,supplied_B=sb,candidate_A=a,candidate_B=b,policies={})
  for policy in ['raw','equal_norm']:
   def ll(h,c):return result['conditions'][h]['candidates'][c]['scores'][policy]['logprob']
   da=ll(sa,a)-ll(sa,b);db=ll(sb,a)-ll(sb,b)
   def tl(h,c):return result['conditions'][h]['candidates'][c]['token_logprobs'][policy]
   contribution=[tl(sa,a)[j]-tl(sa,b)[j]-tl(sb,a)[j]+tl(sb,b)[j] for j in range(10)]
   assert abs(sum(contribution)-(da-db))<1e-10
   entry['policies'][policy]=dict(D_after_A=da,D_after_B=db,E=da-db,delta_logp_A=ll(sa,a)-ll(sb,a),delta_logp_B=ll(sa,b)-ll(sb,b),tokenwise_E_contributions=contribution)
   fa=next(x for x in result['conditions'][sa]['forks'] if x['left']==a and x['right']==b);fb=next(x for x in result['conditions'][sb]['forks'] if x['left']==a and x['right']==b);key=policy+'_margin';fe['policies'][policy]=dict(margin_after_A=fa[key],margin_after_B=fb[key],E=fa[key]-fb[key])
  result['E'].append(entry);result['fork_E'].append(fe)
 result['signs']={policy:dict(negative=sum(x['policies'][policy]['E']<0 for x in result['E']),positive=sum(x['policies'][policy]['E']>0 for x in result['E']),min=min(x['policies'][policy]['E'] for x in result['E']),max=max(x['policies'][policy]['E'] for x in result['E'])) for policy in ['raw','equal_norm']}
 receipts=[read(R/f'runtime/{n}/receipt.json') for n in ['A1','A2','B1','B2']];result['cost']=dict(model_forwards=sum(x['model_forwards'] for x in receipts),vision_forwards=sum(x['vision_forwards'] for x in receipts),allocated_gpu_seconds=sum(x['elapsed_seconds'] for x in receipts),peak_reserved_bytes=max(x['peak_reserved_bytes'] for x in receipts),tensor_bytes=sum(f.stat().st_size for f in (R/'runtime').rglob('*.pt')));assert result['cost']['model_forwards']==176 and result['cost']['allocated_gpu_seconds']<3600 and result['cost']['tensor_bytes']<268435456
 return result
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=pathlib.Path,default=R/'reduction.json');a=ap.parse_args();torch.set_num_threads(4);d=reduce();a.output.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(dict(signs=d['signs'],cost=d['cost'],D=d['D'])))
