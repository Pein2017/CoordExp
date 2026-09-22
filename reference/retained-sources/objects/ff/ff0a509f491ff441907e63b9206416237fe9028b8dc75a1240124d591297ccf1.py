"""Verify frozen execution and same-state shadow evidence using saved tensors only."""
import hashlib,json
from pathlib import Path
import torch
R=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bindcheck(b):
 assert sha(b['path'])==b['sha256'],b['path']
def verify():
 panel=read(R/'panel.json');factor=torch.load(R/'coefficients.pt',weights_only=True);rows=torch.load(R/'effective-readout.pt',weights_only=True)
 assert hashlib.sha256(rows['output_rows'].numpy().tobytes()).hexdigest()==factor['effective_rows_sha256']
 assert torch.equal(rows['input_rows'],rows['output_rows']) and not torch.count_nonzero(rows['bias'])
 records=[];total=0;cost=0;diagnostic_exact=0
 for group in panel['groups']:
  root=R/'runtime'/('group-'+group['key']);rec={k:read(root/k/'receipt.json') for k in ['O','N']};raw={k:read(root/k/'raw.json') for k in rec}
  for k,r in rec.items():
   assert r['status']=='candidate_complete';bindcheck(r['raw']);bindcheck(r['panel']);bindcheck(r['producer']);total+=r['model_forwards'];cost+=r['elapsed_seconds']
  for key in ['input_identity','prefill_mrope_sha256','generate_settings','readout']:assert rec['O'][key]==rec['N'][key]
  for k,field in [('O','rows'),('N','saved_norm_rows')]:
   if field in group:
    for old,current in zip(group[field],raw[k]['rows']):
     assert old.get('token_ids',old.get('generated_token_ids'))==current['token_ids'];assert old.get('stop',old.get('decode_stop_reason'))==current['stop'];diagnostic_exact+=1
  for b,current in enumerate(raw['N']['rows']):
   iid=str(current['image_id']);trace=rec['N']['shadow']['samples'][iid];ids=current['token_ids'];assert trace['token_ids']==ids
   assert len(trace['steps'])==len(ids)
   assert [x['transformed_argmax'] for x in trace['steps']]==ids
   history=[]
   for offset,x in enumerate(trace['steps']):
    assert x['offset']==offset
    assert x['history_sha256']==hashlib.sha256(json.dumps(history,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    assert x['original_top2_margin']>=0 and x['transformed_top2_margin']>=0
    history.append(ids[offset])
   if current['stop']=='im_end':assert ids[-1]==151645 and 151645 not in ids[:-1]
   else:assert current['stop']=='length' and len(ids)==3084 and 151645 not in ids
   disagreements=[x for x in trace['steps'] if x['original_full_vocab_argmax']!=x['transformed_argmax']]
   artifact=rec['N']['shadow']['first_shadow_disagreement_tensors'].get(iid)
   assert bool(artifact)==bool(disagreements)
   if artifact:
    bindcheck(artifact);t=torch.load(artifact['path'],weights_only=True);first=disagreements[0];assert t['offset']==first['offset'];z=t['before_logits'];n=t['after_logits'];assert int(z.argmax())==first['original_full_vocab_argmax'] and int(n.argmax())==first['transformed_argmax']
    assert torch.equal(z[:151670],n[:151670]) and torch.equal(z[152670:],n[152670:])
    assert torch.equal((z[151670:152670].double()*factor['factors']).float(),n[151670:152670])
    reconstructed=rows['output_rows'].double()@t['lm_head_input'].double();err=float((reconstructed-z[151670:152670].double()).abs().max());assert err<.001
    records.append(dict(image_id=int(iid),cohort=group['cohort'],offset=t['offset'],max_abs_readout_error=err,tensor=artifact))
 assert diagnostic_exact==32 and total<=250000 and cost<=36000
 out=dict(status='passed',groups=len(panel['groups']),exact_diagnostic_sequences=diagnostic_exact,model_forwards=total,generation_gpu_seconds=cost,first_shadow_captures=len(records),captures=records,producer=panel['producer'],panel_sha256=sha(R/'panel.json'),no_model_calls=True)
 (R/'saved-verification.json').write_text(json.dumps(out,indent=2)+'\n');print({k:v for k,v in out.items() if k!='captures'})
if __name__=='__main__':verify()
