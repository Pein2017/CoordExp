"""Fixed native groups and output-only norm policy for the untied package."""
import argparse,json,os,time
from pathlib import Path
import torch
from transformers import LogitsProcessor,LogitsProcessorList
from probes.training_set_completion.untied_shared import ROOT,load_model
from probes.training_set_completion.readout_norm_fresh import _binding,_tensor_hash,_input_identity
from src.inference.bound_requests import build_bound_native_requests
from src.qwen.native import prepare_native_inputs
from src.qwen.generation import NativeGenerationPolicy,generate_continuations

def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def run(condition,keys,qualify=False):
 start=time.monotonic();modelkey,policy=condition.split('-');panel=json.loads((ROOT/'panel.json').read_text());q,identity=load_model(modelkey,torch.device('cuda:0'));m=q.model;e=m.get_input_embeddings();h=m.get_output_embeddings();ids=h.selected_token_ids;coords=ids[4:];assert coords.tolist()==list(range(151670,152670))
 E=e(ids).detach();U=h.base.weight[ids].detach()+h.shared_embed_delta.detach();norms=U[4:].double().norm(dim=1);factors=norms.median()/norms;versions={n:v._version for n,v in m.named_parameters()};shared=ROOT/'weights'/condition/keys[0];shared.mkdir(parents=True,exist_ok=True);torch.save(dict(input_rows=E.cpu(),output_rows=U.cpu(),input_delta=e.shared_embed_delta.detach().cpu(),output_delta=h.shared_embed_delta.detach().cpu(),base_rows=h.base.weight[ids].detach().cpu(),selected_ids=ids.cpu(),factors=factors.cpu(),final_norm=m.model.language_model.norm.weight.detach().cpu()),shared/'weights.pt');write(shared/'identity.json',identity)
 for key in keys:
  group=next(g for g in panel['groups'] if g['key']==key);out=ROOT/'runtime'/condition/key;out.mkdir(parents=True,exist_ok=False);rec=dict(status='running',pid=os.getpid(),condition=condition,group=key,model_forwards=0,vision_forwards=0,panel=_binding(ROOT/'panel.json'),producer=_binding(Path(__file__)),identity=identity);write(out/'receipt.json',rec);t=time.monotonic();handles=[]
  try:
   config=dict(panel['configs'][modelkey]);config['data']=dict(input_jsonl=group['input_jsonl']);req,_=build_bound_native_requests(q,config,group['cases']);batch=prepare_native_inputs(q.processor,req,device='cuda:0',record_media_identity=True);rec['input_identity']=_input_identity(batch);width=batch.inputs['input_ids'].shape[1];traces=[];last={}
   def count(module,args,kwargs):rec['model_forwards']+=1;assert rec['model_forwards']<=3300
   def vision(*args):rec['vision_forwards']+=1
   def capture(module,args):last['h']=args[0].detach()
   handles=[m.register_forward_pre_hook(count,with_kwargs=True),m.model.visual.register_forward_pre_hook(vision),h.register_forward_pre_hook(capture)]
   class Policy(LogitsProcessor):
    def __init__(self,mode,record=True):self.mode=mode;self.record=record
    def __call__(self,tokens,scores):
     transformed=scores.clone();transformed[:,coords]=(scores[:,coords].double()*factors).to(scores.dtype);assert torch.equal(transformed[:,:151670],scores[:,:151670]) and torch.equal(transformed[:,152670:],scores[:,152670:]);used=transformed if self.mode=='normalized' else scores
     if self.record:
      top=scores.topk(2);chosen=used.argmax(-1);traces.append(dict(offset=tokens.shape[1]-width,raw_winners=top.indices[:,0].tolist(),raw_top2=top.values.tolist(),raw_runnerups=top.indices[:,1].tolist(),chosen=chosen.tolist(),eos_logits=scores[:,151645].tolist(),logsumexp=scores.logsumexp(-1).tolist(),chosen_raw_logits=scores.gather(1,chosen[:,None]).squeeze(1).tolist()))
     return used
   def generate(mode,budget=3084,record=True):
    original=m.generate
    def wrapped(**kwargs):return original(**kwargs,logits_processor=LogitsProcessorList([] if mode=='plain' else [Policy(mode,record)]))
    m.generate=wrapped
    try:return generate_continuations(m,batch,extensions=[[] for _ in req],budgets=[budget for _ in req],eos_token_id=151645,pad_token_id=q.tokenizer.pad_token_id,policy=NativeGenerationPolicy(temperature=0,top_p=1,top_k=0,repetition_penalty=1,use_model_defaults=False),trace='none',seed=None)
    finally:m.generate=original
   with torch.inference_mode():
    if qualify:
     # Actual multimodal no-op logits, selected row values and independent wrapper paths.
     a=m(**batch.inputs,use_cache=False,logits_to_keep=1).logits.detach();hidden=last['h'];computed=hidden@U.T;assert torch.allclose(a[:,:,ids],computed,atol=.0002,rtol=1e-5)
     explicit=hidden.double()@(U[4:].double()*factors[:,None]).T;assert torch.allclose(explicit,(a[:,:,coords].double()*factors),atol=.0002,rtol=1e-5)
     ones=torch.ones_like(factors);assert torch.equal((a[:,:,coords].double()*ones).to(a.dtype),a[:,:,coords])
     beforeE=e(ids).clone();beforeU=h(hidden).clone();din=e.shared_embed_delta;dout=h.shared_embed_delta
     # Reversible local forward witness; no parameter values survive this gate.
     original_in=din[4,0].item();original_out=dout[5,0].item();din[4,0]+=.01
     changedE=e(ids);changedU=h(hidden);assert not torch.equal(changedE,beforeE)
     if modelkey=='untied':assert torch.equal(changedU,beforeU)
     din[4,0]=original_in;dout[5,0]+=.01;assert not torch.equal(h(hidden),beforeU)
     if modelkey=='untied':assert torch.equal(e(ids),beforeE)
     dout[5,0]=original_out;assert torch.equal(e(ids),beforeE) and torch.equal(h(hidden),beforeU)
     versions={n:v._version for n,v in m.named_parameters()}
     plain=generate('plain',16,False);noop=generate('identity',16,False);assert [list(v.token_ids) for v in plain]==[list(v.token_ids) for v in noop]
     rec['qualification']=dict(effective_row_reconstruction=True,independent_delta_paths=(modelkey=='untied'),temporary_values_exactly_restored=True,no_op_tokens_exact=True,identity_coefficients_exact=True,max_abs=float((a[:,:,ids]-computed).abs().max()))
    values=generate(policy)
   rows=[dict(image_id=int(c['input_record']['image_id']),row_id=c['row_id'],token_ids=list(v.token_ids),text=q.tokenizer.decode(list(v.token_ids),skip_special_tokens=False,clean_up_tokenization_spaces=False),stop=v.stop_reason) for c,v in zip(group['cases'],values)]
   assert versions=={n:v._version for n,v in m.named_parameters()};assert _input_identity(batch)==rec['input_identity'];write(out/'raw.json',dict(rows=rows));write(out/'trace.json',dict(steps=traces));readback=json.loads((out/'raw.json').read_text());assert readback['rows']==rows
   rec.update(status='candidate_complete',raw=_binding(out/'raw.json'),trace=_binding(out/'trace.json'),elapsed_seconds=time.monotonic()-t,active_tokens=sum(len(v['token_ids']) for v in rows),padded_token_work=len(traces)*len(rows),peak_reserved_bytes=torch.cuda.max_memory_reserved(),no_parameter_mutation=True);write(out/'receipt.json',rec)
  except BaseException as exc:rec.update(status='technical_invalid',error=repr(exc),elapsed_seconds=time.monotonic()-t);write(out/'receipt.json',rec);raise
  finally:
   for hook in handles:hook.remove()
  qualify=False
 write(ROOT/f'worker-{condition}-{keys[0]}.json',dict(status='complete',pid=os.getpid(),groups=keys,elapsed_seconds=time.monotonic()-start))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--condition',required=True);a.add_argument('--groups',nargs='+',required=True);a.add_argument('--qualify',action='store_true');x=a.parse_args();run(x.condition,x.groups,x.qualify)
