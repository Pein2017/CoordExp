"""CPU-only independent reconstruction from sealed local tensors and raw outputs."""
import argparse,hashlib,importlib.util,json,pathlib
import torch
R=pathlib.Path(__file__).parent

def bind(p):
 p=pathlib.Path(p);return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),size_bytes=p.stat().st_size)
def read(p):return json.loads(pathlib.Path(p).read_text())
def rank(v,t):return 1+int((v>v[t]).sum())
def top(v):
 x,i=v.topk(2);return dict(token=int(i[0]),margin=float(x[0]-x[1]))
def reduce():
 p=read(R/'panel.json');n=torch.load(R/'runtime/native/capture.pt',map_location='cpu',weights_only=True);r=torch.load(R/'runtime/scores/readout.pt',map_location='cpu',weights_only=True)
 coords=r['coordinate_ids'];W=r['output_rows'].double();norm=r['norms'];f=r['factors'];assert r['bias'] is None and torch.equal(W.norm(dim=1),norm);assert torch.equal(f,norm.median()/norm)
 assert torch.equal(r['output_rows'],r['input_rows']);native=read(R/'runtime/native/raw.json')['rows'];old=read(p['native_raw']['path'])['rows'];assert native==old
 for mode in ['native','scores']:
  rec=read(R/f'runtime/{mode}/receipt.json');assert rec['status']=='candidate_complete' and rec['parameters_unchanged'];assert rec['panel']==bind(R/'panel.json')
 result=dict(windows={},native_parity=[],reconstruction=[],annotation=read(R/'annotation-identity.json'),admission=read(R/'review/admission.json'))
 for window in p['windows']:
  ts=torch.load(R/f"runtime/scores/{window['name']}.pt",map_location='cpu',weights_only=True);out=dict(conditioning=window['policy_history'],row=window['row'],offset=window['offset'],history_sha256=hashlib.sha256(json.dumps(window['history']).encode()).hexdigest(),candidates={},forks=[])
  for name,t in ts.items():
   z=t['logits'].double();h=t['head_input'].double();ids=t['token_ids'];assert len(z)==len(ids)+1 and len(h)==len(z);assert t['history']==window['history']
   recon=h@W.T;error=(recon-z[:,coords]).abs().amax(dim=1);result['reconstruction'].append(dict(window=window['name'],candidate=name,max_abs=float(error.max())))
   sz=z.clone();sz[:,coords]=(z[:,coords]*f).float().double()
   lp=z.log_softmax(-1);slp=sz.log_softmax(-1);tokens=torch.tensor(ids);sel=lp[torch.arange(len(ids)),tokens];sums=float(sel.sum());ci=ids.index(151648)+1
   slots=[]
   for j in range(len(ids)+1):
    field='next_boundary' if j==len(ids) else ['x1','y1','x2','y2'][j-ci] if ci<=j<ci+4 else 'opener' if j==0 else 'description' if 0<j<ids.index(151647) else 'syntax'
    raw=z[j];scaled=sz[j];v=raw[coords];sv=scaled[coords];cos=recon[j]/(norm*h[j].norm());rt=int(v.argmax());st=int(sv.argmax());interior=1+int(v[1:999].argmax());ep=0 if v[0]>=v[999] else 999
    s=dict(row_offset=j,action_offset=window['offset']+j,field=field,raw_top=top(raw),equal_norm_top=top(scaled),eos_logit=float(raw[151645]),eos_rank=rank(raw,151645),opener_logit=float(raw[151646]),opener_rank=rank(raw,151646),coordinate_winner=rt,equal_norm_coordinate_winner=st,cosine_winner=int(cos.argmax()),coordinate_reconstruction_error=float(error[j]),hidden_norm=float(h[j].norm()),winning_row_norm=float(norm[rt]),winning_cosine=float(cos[rt]),strongest_endpoint=ep,strongest_interior=interior,endpoint_minus_interior=float(v[ep]-v[interior]),equal_norm_endpoint_minus_interior=float(sv[ep]-sv[interior]),endpoint_cosine=float(cos[ep]),interior_cosine=float(cos[interior]))
    if j<len(ids):s.update(observed_token=ids[j],observed_rank=rank(raw,ids[j]),observed_logprob=float(sel[j]))
    if field in ['x2','y2']:
     bound=ids[ci+(0 if field=='x2' else 1)]-151670;invalid=coords[:bound+1]
     s['geometry_invalid_mass']=dict(rule=field+' <= preceding '+('x1' if field=='x2' else 'y1'),bound=bound,full_vocabulary=float(lp[j,invalid].exp().sum()),coordinate_conditional=float(v.softmax(0)[:bound+1].sum()),equal_norm_full_vocabulary=float(slp[j,invalid].exp().sum()),equal_norm_coordinate_conditional=float(sv.softmax(0)[:bound+1].sum()))
    slots.append(s)
    # Only identical natural literal histories qualify for native/full parity.
    off=window['offset']+j;hist=window['history']+ids[:j]
    if str(off) in n and hist==n[str(off)]['history']:
     nz=n[str(off)]['logits'].double();eps=float((nz-raw).abs().max());margin=top(nz)['margin'];same=int(nz.argmax())==int(raw.argmax());assert same,(window['name'],name,j,'argmax mismatch')
     # Position identity includes native padding/MRoPE, not merely same token count.
     assert torch.equal(t['positions'][:,-(len(ids)+1)+j],n[str(off)]['positions'][:,-1])
     result['native_parity'].append(dict(window=window['name'],candidate=name,row_offset=j,action_offset=off,max_abs=eps,margin=margin,twice_error_over_margin=2*eps/max(margin,1e-30),same_argmax=same))
   out['candidates'][name]=dict(token_ids=ids,sum_logprob=sums,token_logprobs=sel.tolist(),equal_norm_teacher_forced_sum_logprob=float(slp[torch.arange(len(ids)),tokens].sum()),slots=slots,owner_id=window['candidates'][name].get('owner_id'),support=window['candidates'][name].get('support'),tensor=bind(R/f"runtime/scores/{window['name']}.pt"))
  names=list(ts)
  for a_i,a in enumerate(names):
   for b in names[a_i+1:]:
    aa,bb=ts[a]['token_ids'],ts[b]['token_ids'];j=next((j for j,(x,y) in enumerate(zip(aa,bb)) if x!=y),None)
    if j is None:continue
    v=ts[a]['logits'][j].double();w=ts[b]['logits'][j].double();eps=float((v-w).abs().max());scaled=v.clone();scaled[coords]=(scaled[coords]*f).float().double()
    ia,ib=aa[j],bb[j];margin=float(v[ia]-v[ib]);sm=float(scaled[ia]-scaled[ib]);assert abs(margin)>2*eps,('ambiguous fork',a,b,j,margin,eps)
    terms={}
    if ia in coords and ib in coords:
     h=ts[a]['head_input'][j].double();an,bn=ia-151670,ib-151670;terms=dict(hidden_norm=float(h.norm()),a_row_norm=float(norm[an]),b_row_norm=float(norm[bn]),a_cosine=float((W[an]@h)/(norm[an]*h.norm())),b_cosine=float((W[bn]@h)/(norm[bn]*h.norm())),bias=0)
    out['forks'].append(dict(left=a,right=b,row_offset=j,action_offset=window['offset']+j,left_token=ia,right_token=ib,left_minus_right_margin=margin,equal_norm_left_minus_right_margin=sm,left_rank=rank(v,ia),right_rank=rank(v,ib),equal_norm_left_rank=rank(scaled,ia),equal_norm_right_rank=rank(scaled,ib),full_vocab_raw=top(v),full_vocab_equal_norm=top(scaled),eos_logit=float(v[151645]),eos_rank=rank(v,151645),max_abs_candidate_suffix_numerical_difference=eps,norm_reverses_pair=margin*sm<0,left_minus_right_row_logprob=out['candidates'][a]['sum_logprob']-out['candidates'][b]['sum_logprob'],terms=terms))
  result['windows'][window['name']]=out
 assert result['native_parity'] and max(x['twice_error_over_margin'] for x in result['native_parity'])<1
 # Reuse native parser and frozen one-to-one matcher; separately retain old/current ledgers.
 base=R.parent/'2026-09-16-endpoint-loop-natural-readout-norm';spec=importlib.util.spec_from_file_location('accepted_score',base/'reduce.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 oldpanel=read(base/'panel.json');annpath=R.parent/'2026-09-17-history-rereading-mechanism/human-evaluation/annotation-snapshot-v1/working.norm.jsonl';ann=next(json.loads(l) for l in annpath.open() if json.loads(l)['image_id']==417044)
 bank=[dict(image_id=417044,owner_id=str(o['coco_ann_id']),description=o['desc'],normalized_description=o['desc'].strip().lower(),reference_coord_bins_1000=list(o['bbox_2d'])) for o in ann['objects']]
 result['scoring']={}
 for policy in ['identity','norm']:
  raw=read(base/f"runtime/{p['group']['key']}-{policy}/raw.json")['rows'][p['target']]
  result['scoring'][policy]={label:mod.score(raw,p['group']['cases'][p['target']],bb) for label,bb in [('old17',oldpanel['banks']['417044']),('current63',bank)]}
 result['bank']=bank;result['cost']={k:sum(read(R/f'runtime/{mode}/receipt.json')[k] for mode in ['native','scores']) for k in ['model_forwards','vision_forwards','elapsed_seconds']};assert result['cost']['model_forwards']==3100
 return result
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=pathlib.Path,default=R/'reduction.json');a=ap.parse_args();torch.set_num_threads(4);d=reduce();a.output.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(dict(cost=d['cost'],parity=len(d['native_parity']),rows={k:{n:v['sum_logprob'] for n,v in w['candidates'].items()} for k,w in d['windows'].items()})))
