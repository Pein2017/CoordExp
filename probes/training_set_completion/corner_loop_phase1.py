"""Frozen three-image slot/parity diagnostic; no sampling, updates or intervention."""
from __future__ import annotations
import argparse,copy,hashlib,json,os,time
from pathlib import Path
import torch
from src.config.inference import InferConfig
from probes.dora_owner_learning.runtime import load_policy
from src.inference.bound_requests import build_bound_native_requests
from src.qwen.native import prepare_native_inputs,exact_history_inputs
from probes.owner_successor_scale.replay import _combine_native_inputs

ROOT=Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-16-corner-loop-mechanism')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def thash(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')
def execute(label,iid):
 start=time.monotonic();packet=json.loads((ROOT/'panel.json').read_text());case=next(c for c in packet['cases'] if c['image_id']==iid)
 for b in packet['sources']:assert sha(b['path'])==b['sha256'],b['path']
 out=ROOT/'runtime'/f'{label}-{iid}';out.mkdir(parents=True,exist_ok=False)
 config=copy.deepcopy(packet['config']);config['adapter']['path']=packet['checkpoints'][label]
 qwen,identity=load_policy(InferConfig.model_validate(config),device=torch.device('cuda:0'));model=qwen.model;tok=qwen.tokenizer
 for p in model.parameters():p.requires_grad_(False)
 requests,_=build_bound_native_requests(qwen,config,[case['case']]);batch=prepare_native_inputs(qwen.processor,requests,device='cuda:0',record_media_identity=True)
 prompt=case['prompt_ids'];actions=case['P_action_ids'];assert list(batch.prompt_token_ids[0])==prompt
 assert batch.media_sha256[0]==case['image_identity']['executed_media_sha256']
 assert list(batch.image_grids[0])==case['image_identity']['observed_image_grid_thw']
 receipt={'status':'running','checkpoint':label,'image_id':iid,'panel_sha256':sha(ROOT/'panel.json'),'producer_sha256':sha(__file__),'pid':os.getpid(),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'config':config,'loaded_identity':identity,'counts':{'model_forwards':0,'vision_forwards':0,'new_generated_tokens':0},'boundaries':[],'torch_version':torch.__version__,'batch_shape':packet['batch_shape']}
 write(out/'receipt.json',receipt)
 position=[None]
 def count(*args):receipt['counts']['model_forwards']+=1
 def vision(*args):receipt['counts']['vision_forwards']+=1
 def capture(module,args,kwargs):position[0]=kwargs['position_ids'].detach().clone()
 hs=[model.register_forward_pre_hook(count),model.model.visual.register_forward_pre_hook(vision),model.model.language_model.register_forward_pre_hook(capture,with_kwargs=True)]
 device=torch.device('cuda:0');pad=tok.pad_token_id or 0;opener=tok.convert_tokens_to_ids('<|object_ref_start|>');end=tok.convert_tokens_to_ids('<|object_ref_end|>');bs=tok.convert_tokens_to_ids('<|box_start|>');eos=tok.convert_tokens_to_ids('<|im_end|>')
 coords=[tok.convert_tokens_to_ids(f'<|coord_{i}|>') for i in range(1000)];coord_index={t:i for i,t in enumerate(coords)}
 tensor=lambda ids:torch.tensor([ids],device=device)
 # Use the actual GenerationMixin/Qwen preparation route; Qwen computes cached MRoPE itself.
 def native(ids,cache=None):
  full=tensor(ids);n=cache.get_seq_length() if cache is not None else 0
  kwargs={k:v for k,v in batch.inputs.items() if k not in ['input_ids','attention_mask','position_ids']}
  inputs=model.prepare_inputs_for_generation(full,past_key_values=cache,attention_mask=torch.ones_like(full),cache_position=torch.arange(n,len(ids),device=device),use_cache=True,**kwargs)
  value=model(**inputs,return_dict=True,logits_to_keep=1)
  return value.past_key_values,value.logits[0,-1].float().detach()
 def full(ids,native_inputs=batch.inputs,rows=1,keep=1):
  inputs=exact_history_inputs(model,native_inputs,[ids]*rows,pad_token_id=pad,logits_to_keep=keep)
  value=model(**inputs).logits.float().detach()
  assert torch.equal(position[0],inputs['position_ids'])
  return value,inputs['position_ids']
 def summary(v,ids):
  vals,idx=torch.topk(v,10);chosen=[]
  for token in sorted(set(ids)):
   chosen.append({'id':token,'token':tok.decode([token]),'logit':float(v[token]),'rank':int((v>v[token]).sum())+1,'gap_to_top':float(vals[0]-v[token])})
  return {'argmax':int(idx[0]),'argmax_text':tok.decode([int(idx[0])]),'top_margin':float(vals[0]-vals[1]),'top10':[{'id':int(i),'token':tok.decode([int(i)]),'logit':float(x)} for x,i in zip(vals,idx)],'candidates':chosen}
 with torch.inference_mode():
  # Descriptive identity checks only; no embedding intervention.
  receipt['coordinate_input_sha256']=thash(model.get_input_embeddings()(torch.tensor(coords,device=device)))
  head=model.get_output_embeddings();receipt['readout_type']=type(head).__name__
  # Coordinate embedding deltas can wrap the head, so hash all frozen head parameters.
  receipt['readout_parameter_hashes']={n:thash(p) for n,p in head.named_parameters()}
  cache,current=native(prompt);rope=model.model.rope_deltas.clone();cursor=0
  base_pos=position[0].clone();_,expected_prompt=full(prompt)
  assert torch.equal(base_pos,expected_prompt)
  receipt['prompt_position_sha256']=thash(expected_prompt);receipt['rope_deltas']=rope.cpu().tolist()
  for boundary in case['boundaries']:
   off=boundary['offset'];history=actions[:off]
   # Independent exact derivation for the entire literal P path up to this boundary.
   inputs=exact_history_inputs(model,batch.inputs,[prompt+history],pad_token_id=pad,logits_to_keep=1);expected=inputs['position_ids']
   while cursor<off:
    model.model.rope_deltas=rope.clone();cache,current=native(prompt+actions[:cursor+1],cache)
    assert torch.equal(position[0],expected[:,:,len(prompt)+cursor:len(prompt)+cursor+1]),('position mismatch',cursor)
    cursor+=1
   assert cache.get_seq_length()==len(prompt)+off
   rec={'row_1based':boundary['row_1based'],'history_length':off,'history_sha256':boundary['history_sha256'],'position_sha256':thash(expected),'routes':[]}
   candidates=[('observed',boundary['observed']),('trusted_alternative',boundary['alternative'])]
   if iid==477415 and boundary['row_1based']==138:
    # Same boundary; score the literal preceding corner row as the repeated competitor.
    prev=actions[off-9:off];assert prev[0]==opener
    candidates.append(('repeated_competitor',{'tokens':prev,'text':tok.decode(prev)}))
   for role,row in candidates:
    tokens=row['tokens'];ids=prompt+history+tokens
    recomputed,pos=full(ids,keep=len(tokens)+1);f=recomputed[0,:-1]
    again,_=full(ids,keep=len(tokens)+1);noop=float((again[0,:-1]-f).abs().max())
    native4=_combine_native_inputs([{'inputs':batch.inputs,'prompt_ids':prompt}]*4)
    four,_=full(ids,native_inputs=native4,rows=4,keep=len(tokens)+1);shape_delta=float((four[0,:-1]-f).abs().max())
    branch=copy.deepcopy(cache);v=current.clone();cached=[];rowpositions=[]
    for j,t in enumerate(tokens):
     cached.append(v)
     model.model.rope_deltas=rope.clone();branch,v=native(prompt+history+tokens[:j+1],branch)
     exp=pos[:,:,len(prompt)+off+j:len(prompt)+off+j+1]
     assert torch.equal(position[0],exp),('candidate position mismatch',role,j)
     rowpositions.append(position[0].cpu())
    c=torch.stack(cached)
    if role=='observed':
     clone=copy.deepcopy(cache);model.model.rope_deltas=rope.clone();_,vcheck=native(prompt+history+[tokens[0]],clone)
     assert torch.equal(vcheck,c[1]),'identity cache clone mismatch'
    # Every category token (not only the first) and all coordinate slots, plus admission/EOS.
    cend=tokens.index(end);coordstart=tokens.index(bs)+1
    slots=[('admission',0)]+[(f'category_{j}',j) for j in range(1,cend)]+list(zip(['x1','y1','x2','y2'],range(coordstart,coordstart+4)))
    route={'role':role,'tokens':tokens,'owner_id':row.get('owner_id'),'text':row['text'],'full_noop_max_abs':noop,'bs4_full_max_abs':shape_delta,'cache_full_max_abs':float((c-f).abs().max()),'slots':[]}
    saved={}
    for slot,j in slots:
     eps=float((c[j]-f[j]).abs().max());s=summary(f[j],[tokens[j],eos,opener,coords[0],coords[999],tok.encode('chair',add_special_tokens=False)[0],tok.encode('person',add_special_tokens=False)[0]]+[x['tokens'][x['tokens'].index(bs)+1+['x1','y1','x2','y2'].index(slot)] for _,x in candidates] if slot in ['x1','y1','x2','y2'] else [tokens[j],eos,opener,tok.encode('chair',add_special_tokens=False)[0],tok.encode('person',add_special_tokens=False)[0]])
     cs=summary(c[j],[tokens[j]]);fs4=summary(four[0,j],[tokens[j]])
     material=s['argmax']!=cs['argmax'] and max(s['top_margin'],cs['top_margin'])>max(.001,10*noop)
     route['slots'].append({'slot':slot,'row_token_offset':j,'full':s,'cached':cs,'bs4_full':fs4,'max_abs_cache_full':eps,'twice_error_over_margin':2*eps/max(s['top_margin'],1e-30),'material_parity_failure':material})
     saved[slot+'.full']=f[j].cpu();saved[slot+'.cache']=c[j].cpu();saved[slot+'.bs4']=four[0,j].cpu()
    torch.save(saved,out/f"row{boundary['row_1based']}-{role}-logits.pt")
    rec['routes'].append(route)
    del branch,c,f,recomputed,again,four,saved
   receipt['boundaries'].append(rec);write(out/'receipt.json',receipt)
   if any(s['material_parity_failure'] for rr in rec['routes'] for s in rr['slots']):
    receipt['status']='blocked_material_execution_parity';break
  else:receipt['status']='candidate_complete'
  torch.cuda.synchronize();receipt['elapsed_seconds']=time.monotonic()-start;receipt['peak_allocated_bytes']=torch.cuda.max_memory_allocated();receipt['peak_reserved_bytes']=torch.cuda.max_memory_reserved()
 for h in hs:h.remove()
 write(out/'receipt.json',receipt)
 print(json.dumps({'status':receipt['status'],'cell':out.name,'seconds':receipt['elapsed_seconds'],'counts':receipt['counts']}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('checkpoint',choices=['Bnormalized64','P16','R16']);p.add_argument('image',type=int,choices=[477415,351017,417044]);a=p.parse_args();execute(a.checkpoint,a.image)
