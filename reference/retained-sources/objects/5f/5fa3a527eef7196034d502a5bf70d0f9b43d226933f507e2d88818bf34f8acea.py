"""One still-invalid y2 edit through the accepted original heterogeneous native batch."""
from __future__ import annotations
import argparse,hashlib,json,os,time,traceback
from pathlib import Path
import torch
from transformers import LogitsProcessor,LogitsProcessorList
from src.config.inference import InferConfig
from src.inference.bound_requests import build_bound_native_requests
from src.qwen.native import prepare_native_inputs
from src.qwen.generation import generate_continuations,NativeGenerationPolicy
from probes.dora_owner_learning.runtime import load_policy
ROOT=Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-16-corner-loop-one-bin-control')
def bind(p):return {'path':str(p),'sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest(),'size_bytes':Path(p).stat().st_size}
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')
def execute(cell):
 start=time.monotonic();panel=json.loads((ROOT/'panel.json').read_text());out=ROOT/'runtime'/cell;out.mkdir(parents=True,exist_ok=False)
 receipt={'schema':'corner_loop.bridge_factorial.cell.v1','status':'running','cell':cell,'panel':bind(ROOT/'panel.json'),'producer':bind(Path(__file__)),'pid':os.getpid(),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'counts':{'model_forwards':0,'vision_forwards':0,'processor_calls':0},'edits':panel['cells'][cell]['edits'],'seam_logits':[],'positions':[],'mechanical_forks':[]}
 write(out/'receipt.json',receipt)
 try:
  for b in panel['sources']:assert bind(b['path'])==b,b['path']
  if cell!='C00':
   control=json.loads((Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-16-corner-loop-bridge-factorial/runtime/C00/receipt.json')).read_text());assert control['status']=='candidate_complete' and control['full_original_batch_identity']
   receipt['control_gate']=bind(Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-16-corner-loop-bridge-factorial/runtime/C00/receipt.json'))
  qwen,identity=load_policy(InferConfig.model_validate(panel['config']),device=torch.device('cuda:0'));model=qwen.model;tok=qwen.tokenizer;receipt['loaded_identity']=identity
  requests,_=build_bound_native_requests(qwen,panel['config'],panel['cases']);batch=prepare_native_inputs(qwen.processor,requests,device='cuda:0',record_media_identity=True)
  group=panel['batch_group'];assert [list(x) for x in batch.prompt_token_ids]==[g['prompt_token_ids'] for g in group]
  assert list(batch.media_sha256)==[g['executed_media_sha256'] for g in group]
  assert [list(x) for x in batch.image_grids]==[g['observed_image_grid_thw'] for g in group]
  width=batch.inputs['input_ids'].shape[1];assert width==1362
  receipt['batch_shape']={'size':4,'image_ids':[g['image_id'] for g in group],'target_position':1,'padded_prompt_width':width,'left_padding':[width-len(g['prompt_token_ids']) for g in group],'pad_token_id':tok.pad_token_id}
  expected=panel['cells'][cell]['history_ids'];original=group[1]['generated_token_ids'];prompt_last=[None];first_positions=[None]
  def model_count(module,args,kwargs):
   receipt['counts']['model_forwards']+=1
   assert receipt['counts']['model_forwards']<=5000,'per-cell ceiling'
   if receipt['counts']['model_forwards'] in [1,1225,1234]:
    cache=kwargs.get('past_key_values');receipt['positions'].append({'call':receipt['counts']['model_forwards'],'input_width':kwargs['input_ids'].shape[1],'cache_length_before':cache.get_seq_length() if cache is not None else 0,'cache_position':kwargs['cache_position'].cpu().tolist()})
  def vision(*args):receipt['counts']['vision_forwards']+=1
  def positions(module,args,kwargs):
   pos=kwargs['position_ids'];step=receipt['counts']['model_forwards']-1
   if step==0:
    prompt_last[0]=pos[:,:,-1:].clone();first_positions[0]=pos.cpu();receipt['prompt_positions_sha256']=hashlib.sha256(pos.cpu().numpy().tobytes()).hexdigest()
   else:assert torch.equal(pos,prompt_last[0]+step),'native incremental MRoPE differs'
  handles=[model.register_forward_pre_hook(model_count,with_kwargs=True),model.model.visual.register_forward_pre_hook(vision),model.model.language_model.register_forward_pre_hook(positions,with_kwargs=True)]
  class SuppliedRow(LogitsProcessor):
   def __call__(self,ids,scores):
    offset=ids.shape[1]-width;receipt['counts']['processor_calls']+=1
    # Companion requests are exact production companions, not new scientific cases.
    for b,g in enumerate(group):
     if b==1:continue
     tokens=g['generated_token_ids']
     if offset<len(tokens) and int(scores[b].argmax())!=tokens[offset]:
      receipt['mechanical_forks'].append({'batch_position':b,'action_offset':offset,'expected':tokens[offset],'actual':int(scores[b].argmax())});torch.save(scores.cpu(),out/'fork-logits.pt');write(out/'receipt.json',receipt);raise RuntimeError('companion replay mismatch')
    must_match=offset<1224 or cell=='C00'
    v=scores[1];top=torch.topk(v,5)
    if must_match and offset<len(original) and int(top.indices[0])!=original[offset]:
     receipt['mechanical_forks'].append({'batch_position':1,'action_offset':offset,'expected':original[offset],'actual':int(top.indices[0]),'argmax_margin':float(top.values[0]-top.values[1]),'expected_gap':float(top.values[0]-v[original[offset]])});torch.save(scores.cpu(),out/'fork-logits.pt');torch.save(ids.cpu(),out/'fork-history.pt');write(out/'receipt.json',receipt);raise RuntimeError('untouched native target replay mismatch')
    if 1224<=offset<1245:
     receipt['seam_logits'].append({'action_offset':offset,'top5':[{'id':int(i),'text':tok.decode([int(i)]),'logit':float(x)} for x,i in zip(top.values,top.indices)],'original_token':original[offset],'original_logit':float(v[original[offset]]),'supplied_token':expected[offset] if offset<1233 else None})
    if cell!='C00' and 1224<=offset<1233:
     scores=scores.clone();scores[1,:]=-torch.inf;scores[1,expected[offset]]=0
    return scores
  original_generate=model.generate
  def with_supplied_row(**kwargs):
   assert kwargs['max_new_tokens']==3084 and kwargs['repetition_penalty']==1 and not kwargs['do_sample']
   receipt['generate_settings']={k:kwargs[k] for k in ['max_new_tokens','do_sample','repetition_penalty','eos_token_id','pad_token_id','use_model_defaults','return_dict_in_generate','output_scores','output_logits']}
   assert 'logits_processor' not in kwargs
   return original_generate(**kwargs,logits_processor=LogitsProcessorList([SuppliedRow()]))
  model.generate=with_supplied_row
  try:
   result=generate_continuations(model,batch,extensions=[[]]*4,budgets=[3084]*4,eos_token_id=151645,pad_token_id=tok.pad_token_id,policy=NativeGenerationPolicy(temperature=0,top_p=1,top_k=0,repetition_penalty=1,use_model_defaults=False),trace='none',seed=None)
  finally:
   model.generate=original_generate
   for h in handles:h.remove()
  rows=[]
  for b,item in enumerate(result):
   ids=list(item.token_ids);rows.append({'image_id':group[b]['image_id'],'token_ids':ids,'tokens_sha256':digest(ids),'stop':item.stop_reason,'text':tok.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)})
   if b!=1 or cell=='C00':assert ids==group[b]['generated_token_ids'] and item.stop_reason==group[b]['decode_stop_reason']
  target=rows[1];assert target['token_ids'][:1233]==expected
  free=target['token_ids'][1233:];assert 0<len(free)<=1851
  assert (target['stop']=='im_end' and free[-1]==151645) or (target['stop']=='length' and len(free)==1851 and 151645 not in free)
  receipt.update(status='candidate_complete',full_original_batch_identity=cell=='C00',companion_identity=True,common136_identity=True,supplied137_identity=True,free_tokens=len(free),stop=target['stop'],target_total_tokens=len(target['token_ids']),elapsed_seconds=time.monotonic()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
  write(out/'raw.json',{'cell':cell,'rows':rows,'free_token_ids':free,'common136_token_ids':panel['common_history_ids'],'supplied137_token_ids':panel['cells'][cell]['tokens'],'free_text':tok.decode(free,skip_special_tokens=False,clean_up_tokenization_spaces=False),'policy':'native greedy freely after completed row137; no next opener forced; EOS available'})
  receipt['raw']=bind(out/'raw.json');write(out/'receipt.json',receipt);print(json.dumps({k:receipt[k] for k in ['status','cell','counts','free_tokens','stop','elapsed_seconds']}))
 except Exception as e:
  receipt.update(status='blocked_technical',error=repr(e),elapsed_seconds=time.monotonic()-start);write(out/'receipt.json',receipt);raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('cell',choices=['Y998']);execute(p.parse_args().cell)
