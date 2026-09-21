import argparse,copy,hashlib,inspect,json,os,time,traceback
from pathlib import Path
import torch
from src.config.inference import InferConfig
from probes.dora_owner_learning.runtime import load_policy
from src.inference.bound_requests import build_bound_native_requests
from src.qwen.native import prepare_native_inputs,exact_history_inputs
from src.qwen.special_token_embeddings import SelectedDeltaOutputHead,SelectedDeltaInputEmbedding
R=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
def bind(p):return dict(path=str(p),sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())
def th(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def execute(label,iid):
 start=time.monotonic();p=read(R/'panel.json');c=next(x for x in p['cells'] if x['image_id']==iid);out=R/'runtime'/f'{label}-{iid}';out.mkdir(parents=True,exist_ok=False)
 rec=dict(status='running',pid=os.getpid(),cell=out.name,panel=bind(R/'panel.json'),producer=bind(__file__),forwards=0,vision_calls=0,generated_tokens=0,rows=[]);write(out/'receipt.json',rec)
 try:
  for b in p['sources']:assert bind(b['path'])['sha256']==b['sha256'],b['path']
  cfg=copy.deepcopy(p['config']);cfg['adapter']['path']=p['checkpoints'][label];q,identity=load_policy(InferConfig.model_validate(cfg),device=torch.device('cuda:0'));m=q.model;rec['loaded_identity']=identity
  for x in m.parameters():x.requires_grad_(False)
  req,_=build_bound_native_requests(q,cfg,[c['case']['case']]);batch=prepare_native_inputs(q.processor,req,device='cuda:0',record_media_identity=True);prompt=c['case']['prompt_ids'];assert list(batch.prompt_token_ids[0])==prompt;assert batch.media_sha256[0]==c['case']['image_identity']['executed_media_sha256'];assert list(batch.image_grids[0])==c['case']['image_identity']['observed_image_grid_thw']
  head=m.get_output_embeddings();emb=m.get_input_embeddings();assert isinstance(head,SelectedDeltaOutputHead) and isinstance(emb,SelectedDeltaInputEmbedding)
  coords=torch.tensor(p['coordinate_ids'],device='cuda:0');assert [q.tokenizer.convert_tokens_to_ids(f'<|coord_{i}|>') for i in range(1000)]==p['coordinate_ids']
  lookup={int(t):i for i,t in enumerate(head.selected_token_ids.tolist())};drows=torch.tensor([lookup[int(t)] for t in coords],device='cuda:0');base=head.base.weight[coords].detach();delta=head.shared_embed_delta[drows].detach();w=base+delta;inp=emb(coords).detach();bias=head.bias[coords].detach() if head.bias is not None else torch.zeros(1000,device='cuda:0')
  assert torch.equal(w,inp);assert emb.base.weight.data_ptr()==head.base.weight.data_ptr();assert emb.shared_embed_delta.data_ptr()==head.shared_embed_delta.data_ptr()
  rec['readout']=dict(type=type(head).__name__,bias_exists=head.bias is not None,tied_base=True,shared_delta=True,input_effective_sha256=th(inp),output_effective_sha256=th(w),base_sha256=th(base),delta_sha256=th(delta),head_source=bind(inspect.getfile(type(head))),model_source=bind(inspect.getfile(type(m))),language_source=bind(inspect.getfile(type(m.model.language_model))),coordinate_ids=p['coordinate_ids'])
  tensors=dict(output_rows=w.cpu(),input_rows=inp.cpu(),base_rows=base.cpu(),delta_rows=delta.cpu(),bias=bias.cpu());captured=[None];pos=[None]
  def capture(module,args):captured[0]=args[0].detach()
  def position(module,args,kwargs):pos[0]=kwargs['position_ids'].detach()
  def count(*args):rec['forwards']+=1;assert rec['forwards']<=len(c['rows'])
  def vision(*args):rec['vision_calls']+=1
  hooks=[head.register_forward_pre_hook(capture),m.model.language_model.register_forward_pre_hook(position,with_kwargs=True),m.register_forward_pre_hook(count),m.model.visual.register_forward_pre_hook(vision)]
  with torch.inference_mode():
   for row in c['rows']:
    ids=prompt+row['history']+row['tokens'];inputs=exact_history_inputs(m,batch.inputs,[ids],pad_token_id=q.tokenizer.pad_token_id or 0,logits_to_keep=len(row['tokens'])+1);native=m(**inputs).logits[0,:-1].float();h=captured[0][0,:-1].float();assert torch.equal(pos[0],inputs['position_ids'])
    j=row['tokens'].index(151648)+1;indices=list(range(j,j+4));H=h[indices];Z=native[indices][:,coords];recon=H.double()@w.double().T+bias.double();err=(recon-Z.double()).abs().amax(1);margin=Z.topk(2).values.diff(dim=1).abs()[:,0]
    assert torch.equal(recon.argmax(1),Z.argmax(1)),'affine argmax mismatch';assert bool((2*err<margin).all()),'margin-relevant reconstruction failure'
    saved_error=None
    if row['saved_row'] is not None:
     path=R.parent/'2026-09-16-corner-loop-mechanism'/f"runtime/{label}-{iid}/row{row['saved_row']}-observed-logits.pt";old=torch.load(path,map_location='cpu',weights_only=True);oldz=torch.stack([old[s+'.full'][p['coordinate_ids']] for s in ['x1','y1','x2','y2']]).to(Z.device);saved_error=float((Z-oldz).abs().max());assert torch.equal(Z.argmax(1),oldz.argmax(1))
    tensors[row['key']+'.hidden']=H.cpu();tensors[row['key']+'.logits']=Z.cpu()
    rec['rows'].append(dict(key=row['key'],stage=row['stage'],origin=row['origin'],history_tokens=len(row['history']),input_tokens=len(ids),position_sha256=th(pos[0]),hidden_sha256=th(H),logits_sha256=th(Z),affine_max_abs=err.cpu().tolist(),native_margins=margin.cpu().tolist(),saved_logits_max_abs=saved_error,coordinate_positions=indices,native_full_vocab_argmax=native[indices].argmax(1).cpu().tolist()))
    del native,h,H,Z,recon,inputs
   for hook in hooks:hook.remove()
  torch.save(tensors,out/'tensors.pt');torch.cuda.synchronize();rec.update(status='candidate_complete',tensors=bind(out/'tensors.pt'),elapsed_seconds=time.monotonic()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved());assert rec['peak_reserved_bytes']<=p['bounds']['peak_memory_per_gpu_bytes'];write(out/'receipt.json',rec);print(rec['cell'],rec['status'],rec['forwards'],rec['elapsed_seconds'])
 except Exception as exc:
  rec.update(status='technical_invalid',error=repr(exc),elapsed_seconds=time.monotonic()-start);write(out/'receipt.json',rec);raise
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('checkpoint');a.add_argument('image',type=int);x=a.parse_args();execute(x.checkpoint,x.image)
