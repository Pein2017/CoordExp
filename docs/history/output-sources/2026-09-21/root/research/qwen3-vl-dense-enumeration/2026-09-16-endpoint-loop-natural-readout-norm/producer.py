import argparse,hashlib,json,os,time
from pathlib import Path
import torch
from transformers import LogitsProcessor,LogitsProcessorList
from src.config.inference import InferConfig
from probes.dora_owner_learning.runtime import load_policy
from src.inference.bound_requests import build_bound_native_requests
from src.qwen.native import prepare_native_inputs
from src.qwen.generation import generate_continuations,NativeGenerationPolicy
from src.qwen.special_token_embeddings import SelectedDeltaOutputHead,SelectedDeltaInputEmbedding
R=Path(__file__).resolve().parent;PRIOR=R.parent/'2026-09-16-endpoint-loop-readout-state'
read=lambda p:json.loads(p.read_text())
def bind(p):return dict(path=str(p),sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest(),size_bytes=Path(p).stat().st_size)
def th(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def write(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
def field(ids,expected):
 if expected==151645:return 'EOS'
 a=max([i for i,x in enumerate(ids) if x==151648],default=-1);e=max([i for i,x in enumerate(ids) if x==151649],default=-1)
 if a>e and 0<=len(ids)-a-1<4:return ['x1','y1','x2','y2'][len(ids)-a-1]
 if expected==151646:return 'row_opener'
 if expected in [151647,151648,151649]:return 'syntax_delimiter'
 return 'category_or_other'
def execute(key,policy):
 start=time.monotonic();p=read(R/'panel.json');g=next(x for x in p['groups'] if x['key']==key);out=R/'runtime'/f'{key}-{policy}';out.mkdir(parents=True,exist_ok=False);rec=dict(status='running',pid=os.getpid(),group=key,policy=policy,panel=bind(R/'panel.json'),producer=bind(__file__),model_forwards=0,vision_forwards=0,processor_calls=0,first_forks={},offline_checks=[],positions=[]);write(out/'receipt.json',rec)
 try:
  for b in p['sources']:assert bind(b['path'])==b,b['path']
  assert bind(R/'coefficients.pt')==p['coefficients']
  if policy=='norm':
   control=read(R/f'runtime/{key}-identity/receipt.json');assert control['status']=='candidate_complete' and control['saved_batch_exact'];rec['identity_gate']=bind(R/f'runtime/{key}-identity/receipt.json')
  q,ident=load_policy(InferConfig.model_validate(p['config']),device=torch.device('cuda:0'));m=q.model;tok=q.tokenizer;rec['loaded_identity']=ident
  req,_=build_bound_native_requests(q,p['config'],g['cases']);batch=prepare_native_inputs(q.processor,req,device='cuda:0',record_media_identity=True)
  assert [list(x) for x in batch.prompt_token_ids]==[x['prompt_token_ids'] for x in g['rows']];assert list(batch.media_sha256)==[x['executed_media_sha256'] for x in g['rows']];assert [list(x) for x in batch.image_grids]==[x['observed_image_grid_thw'] for x in g['rows']]
  width=batch.inputs['input_ids'].shape[1];rec['batch_shape']=dict(size=4,image_ids=[x['image_id'] for x in g['rows']],prompt_width=width,left_padding=[width-len(x['prompt_token_ids']) for x in g['rows']],attention_mask_sha256=th(batch.inputs['attention_mask']),input_ids_sha256=th(batch.inputs['input_ids']))
  head=m.get_output_embeddings();emb=m.get_input_embeddings();assert isinstance(head,SelectedDeltaOutputHead) and isinstance(emb,SelectedDeltaInputEmbedding) and head.bias is None
  coords=torch.tensor(p['coordinate_ids'],device='cuda:0');assert coords.tolist()==[tok.convert_tokens_to_ids(f'<|coord_{i}|>') for i in range(1000)]
  ix={int(t):i for i,t in enumerate(head.selected_token_ids.tolist())};di=torch.tensor([ix[int(x)] for x in coords],device='cuda:0');effective=head.base.weight[coords].detach()+head.shared_embed_delta[di].detach();assert torch.equal(effective,emb(coords).detach());coeff=torch.load(R/'coefficients.pt',map_location='cpu',weights_only=True);assert th(effective)==coeff['effective_rows_sha256'];n=effective.cpu().double().norm(dim=1);assert torch.equal(n,coeff['norms']) and torch.equal(n.median()/n,coeff['factors']);factors=coeff['factors'].to('cuda:0')
  versions={name:x._version for name,x in m.named_parameters()};input_before=th(emb(coords));base_before=th(head.base.weight[coords]);delta_before=th(head.shared_embed_delta);rec['readout']=dict(effective_sha256=th(effective),input_sha256=input_before,base_sha256=base_before,delta_sha256=delta_before,bias_exists=False,tied_base=head.base.weight.data_ptr()==emb.base.weight.data_ptr(),shared_delta=head.shared_embed_delta.data_ptr()==emb.shared_embed_delta.data_ptr(),coefficients=bind(R/'coefficients.pt'))
  prior={str(i):torch.load(PRIOR/f'runtime/R16-{i}/tensors.pt',map_location='cpu',weights_only=True) for i in g['focus_ids'] if str(i) in p['prior_first_rows']};prompt_last=[None]
  def count(module,args,kwargs):
   rec['model_forwards']+=1;assert rec['model_forwards']<=3084
   if rec['model_forwards'] in [1,2,100]:rec['positions'].append(dict(call=rec['model_forwards'],cache_position=kwargs['cache_position'].cpu().tolist(),input_width=kwargs['input_ids'].shape[1]))
  def vision(*args):rec['vision_forwards']+=1
  def position(module,args,kwargs):
   pos=kwargs['position_ids'];step=rec['model_forwards']-1
   if step==0:prompt_last[0]=pos[:,:,-1:].clone();rec['prompt_position_sha256']=th(pos)
   else:assert torch.equal(pos,prompt_last[0]+step)
  hooks=[m.register_forward_pre_hook(count,with_kwargs=True),m.model.visual.register_forward_pre_hook(vision),m.model.language_model.register_forward_pre_hook(position,with_kwargs=True)]
  class NormPolicy(LogitsProcessor):
   def __call__(self,ids,scores):
    offset=ids.shape[1]-width;rec['processor_calls']+=1;scaled=scores.clone();scaled[:,coords]=(scores[:,coords].double()*factors).to(scores.dtype)
    assert torch.equal(scores[:,:151670],scaled[:,:151670]) and torch.equal(scores[:,152670:],scaled[:,152670:]);assert torch.equal(scores[:,151645],scaled[:,151645]);assert torch.isfinite(scaled).all()
    if offset==0:rec['seam_check']=dict(noncoordinate_bitwise_unchanged=True,eos_bitwise_unchanged=True,all1000_coordinates_scaled=True,all4_samples=True,all_steps=True,coefficient_sha256=th(factors),formula_max_abs=float((scaled[:,coords].double()-scores[:,coords].double()*factors).abs().max()))
    output=scaled if policy=='norm' else scores
    for b,row in enumerate(g['rows']):
     original=row['generated_token_ids'];history=ids[b,width:].tolist();expected=original[offset] if offset<len(original) else None
     if policy=='identity' and offset<len(original):assert int(scores[b].argmax())==expected,('identity fork',row['image_id'],offset)
     keyid=str(row['image_id'])
     if policy=='norm' and keyid not in rec['first_forks'] and offset<len(original) and int(output[b].argmax())!=expected:
      assert history==original[:offset];path=out/f'first-fork-{keyid}.pt';torch.save(dict(before=scores[b].cpu(),after=scaled[b].cpu()),path)
      def top(v):
       val,ind=v.topk(5);return [dict(token_id=int(i),text=tok.decode([int(i)]),logit=float(x)) for x,i in zip(val,ind)]
      rec['first_forks'][keyid]=dict(offset=offset,field=field(history,expected),original_token=expected,native_before_token=int(scores[b].argmax()),treated_token=int(output[b].argmax()),before_top5=top(scores[b]),after_top5=top(output[b]),eos_before=float(scores[b,151645]),eos_after=float(output[b,151645]),same_literal_prefix=True,history_sha256=hashlib.sha256(json.dumps(history).encode()).hexdigest(),tensor=bind(path))
     if keyid in prior:
      literal=p['prior_first_rows'][keyid]['tokens'];ci=literal.index(151648)+1
      if ci<=offset<ci+4 and history==literal[:offset]:
       j=offset-ci;z=prior[keyid]['P-row1.logits'][j].to(scores.device).double();oldscaled=z*factors;current=scaled[b,coords].double();error=float((current-oldscaled).abs().max());margin=float(oldscaled.topk(2).values.diff().abs()[0]);assert int(current.argmax())==int(oldscaled.argmax()) and 2*error<margin
       rec['offline_checks'].append(dict(image_id=row['image_id'],slot=['x1','y1','x2','y2'][j],offset=offset,same_prefix=True,max_abs_scaled_error=error,twice_error_over_margin=2*error/margin,prior_coordinate_argmax=int(oldscaled.argmax()),runtime_coordinate_argmax=int(current.argmax()),prior_tensor=bind(PRIOR/f'runtime/R16-{keyid}/tensors.pt')))
    return output
  original_generate=m.generate
  def wrap(**kwargs):
   assert kwargs['max_new_tokens']==3084 and kwargs['repetition_penalty']==1 and not kwargs['do_sample'];assert 'logits_processor' not in kwargs
   rec['generate_settings']={k:kwargs[k] for k in ['max_new_tokens','do_sample','repetition_penalty','eos_token_id','pad_token_id','use_model_defaults','return_dict_in_generate','output_scores','output_logits']}
   return original_generate(**kwargs,logits_processor=LogitsProcessorList([NormPolicy()]))
  m.generate=wrap
  try:
   with torch.inference_mode():values=generate_continuations(m,batch,extensions=[[]]*4,budgets=[3084]*4,eos_token_id=151645,pad_token_id=tok.pad_token_id,policy=NativeGenerationPolicy(temperature=0,top_p=1,top_k=0,repetition_penalty=1,use_model_defaults=False),trace='none',seed=None)
  finally:
   m.generate=original_generate
   for hook in hooks:hook.remove()
  assert versions=={name:x._version for name,x in m.named_parameters()};assert th(emb(coords))==input_before and th(head.base.weight[coords])==base_before and th(head.shared_embed_delta)==delta_before
  rows=[]
  for old,v in zip(g['rows'],values):
   ids=list(v.token_ids);assert len(ids)<=3084;rows.append(dict(image_id=old['image_id'],token_ids=ids,text=tok.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False),stop=v.stop_reason))
   if policy=='identity':assert ids==old['generated_token_ids'] and v.stop_reason==old['decode_stop_reason']
  write(out/'raw.json',dict(group=key,policy=policy,rows=rows,empty_prefix=True));torch.cuda.synchronize();rec.update(status='candidate_complete',saved_batch_exact=policy=='identity',no_parameter_mutation=True,parameter_version_count=len(versions),raw=bind(out/'raw.json'),elapsed_seconds=time.monotonic()-start,peak_reserved_bytes=torch.cuda.max_memory_reserved());write(out/'receipt.json',rec);print(key,policy,rec['model_forwards'],rec['elapsed_seconds'])
 except Exception as exc:
  rec.update(status='technical_invalid',error=repr(exc),elapsed_seconds=time.monotonic()-start);write(out/'receipt.json',rec);raise
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('group');a.add_argument('policy',choices=['identity','norm']);x=a.parse_args();execute(x.group,x.policy)
