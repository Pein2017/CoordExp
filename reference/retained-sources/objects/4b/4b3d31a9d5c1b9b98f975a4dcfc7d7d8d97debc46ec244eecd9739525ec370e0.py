"""Frozen owner-row scores and untouched production-shaped native replay."""
from __future__ import annotations
import argparse, copy, json, os, time
from pathlib import Path
import torch
from transformers import LogitsProcessor, LogitsProcessorList
from probes.dora_owner_learning.runtime import load_policy
from probes.training_set_completion.readout_norm_fresh import _binding, _input_identity, _tensor_hash, _write
from src.config.inference import InferConfig
from src.inference.bound_requests import build_bound_native_requests
from src.qwen.native import prepare_native_inputs, exact_history_inputs
from src.qwen.generation import generate_continuations, NativeGenerationPolicy
from src.qwen.special_token_embeddings import SelectedDeltaOutputHead


def execute(root: Path, mode: str):
    start=time.monotonic(); p=json.loads((root/'panel.json').read_text()); g=p['group']; target=p['target']
    out=root/'runtime'/mode; out.mkdir(parents=True,exist_ok=False)
    rec=dict(status='running',pid=os.getpid(),mode=mode,panel=_binding(root/'panel.json'),producer=_binding(Path(__file__)),model_forwards=0,vision_forwards=0,cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    _write(out/'receipt.json',rec)
    try:
        for b in p['sources']: assert _binding(Path(b['path']))==b,b['path']
        q,ident=load_policy(InferConfig.model_validate(p['config']),device=torch.device('cuda:0'));m=q.model.eval();tok=q.tokenizer
        req,_=build_bound_native_requests(q,p['config'],g['cases']);batch=prepare_native_inputs(q.processor,req,device='cuda:0',record_media_identity=True)
        assert [list(x) for x in batch.prompt_token_ids]==[x['prompt_token_ids'] for x in g['rows']]
        assert list(batch.media_sha256)==[x['executed_media_sha256'] for x in g['rows']]
        assert [list(x) for x in batch.image_grids]==[x['observed_image_grid_thw'] for x in g['rows']]
        oldrec=json.loads(Path(p['native_receipt']['path']).read_text());assert ident==oldrec['loaded_identity']
        width=batch.inputs['input_ids'].shape[1];assert _tensor_hash(batch.inputs['input_ids'])==oldrec['batch_shape']['input_ids_sha256'];assert _tensor_hash(batch.inputs['attention_mask'])==oldrec['batch_shape']['attention_mask_sha256']
        rec.update(loaded_identity=ident,input_identity=_input_identity(batch),batch_shape=oldrec['batch_shape'])
        head=m.get_output_embeddings();assert isinstance(head,SelectedDeltaOutputHead) and head.bias is None
        coords=torch.tensor(p['coordinate_ids'],device='cuda:0');assert coords.tolist()==[tok.convert_tokens_to_ids(f'<|coord_{i}|>') for i in range(1000)]
        lookup={int(t):i for i,t in enumerate(head.selected_token_ids.tolist())};di=torch.tensor([lookup[int(t)] for t in coords],device='cuda:0')
        W=head.base.weight[coords].detach()+head.shared_embed_delta[di].detach();E=m.get_input_embeddings()(coords).detach();assert torch.equal(W,E)
        coeff=torch.load(p['coefficients']['path'],map_location='cpu',weights_only=True);assert _tensor_hash(W)==coeff['effective_rows_sha256'];assert torch.equal(W.cpu().double().norm(dim=1),coeff['norms'])
        torch.save(dict(output_rows=W.cpu(),input_rows=E.cpu(),coordinate_ids=coords.cpu(),norms=coeff['norms'],factors=coeff['factors'],bias=None),out/'readout.pt')
        rec['readout']=dict(effective_sha256=_tensor_hash(W),bias_exists=False,tied_base=head.base.weight.data_ptr()==m.get_input_embeddings().base.weight.data_ptr(),shared_delta=head.shared_embed_delta.data_ptr()==m.get_input_embeddings().shared_embed_delta.data_ptr())
        versions={n:v._version for n,v in m.named_parameters()};captured={};last={};active=[mode=='scores']
        def count(module,args,kwargs):
            rec['model_forwards']+=1;assert rec['model_forwards']<=(3084 if mode=='native' else 16)
            if mode=='native':active[0]=rec['model_forwards']-1 in p['native_capture_offsets']
        def vision(*args):rec['vision_forwards']+=1
        def pos(module,args,kwargs):
            if active[0]:last['positions']=kwargs['position_ids'][:,target].detach().cpu().clone()
        def hhook(module,args):
            if active[0]:last['head_input']=args[0][target].detach().cpu().clone()
        hooks=[m.register_forward_pre_hook(count,with_kwargs=True),m.model.visual.register_forward_pre_hook(vision),m.model.language_model.register_forward_pre_hook(pos,with_kwargs=True),head.register_forward_pre_hook(hhook)]
        saved=json.loads(Path(p['native_raw']['path']).read_text())['rows']
        with torch.inference_mode():
            if mode=='native':
                class Observe(LogitsProcessor):
                    def __call__(self,ids,scores):
                        off=ids.shape[1]-width
                        for j,row in enumerate(saved):
                            if off<len(row['token_ids']):assert int(scores[j].argmax())==row['token_ids'][off],(j,off)
                        if active[0]:captured[str(off)]=dict(logits=scores[target].detach().cpu().clone(),head_input=last['head_input'][-1].clone(),positions=last['positions'].clone(),history=ids[target,width:].tolist())
                        return scores
                original=m.generate
                def wrap(**kwargs):
                    assert kwargs['max_new_tokens']==3084 and kwargs['repetition_penalty']==1 and not kwargs['do_sample']
                    return original(**kwargs,logits_processor=LogitsProcessorList([Observe()]))
                m.generate=wrap
                try: values=generate_continuations(m,batch,extensions=[[]]*4,budgets=[3084]*4,eos_token_id=151645,pad_token_id=tok.pad_token_id,policy=NativeGenerationPolicy(temperature=0,top_p=1,top_k=0,repetition_penalty=1,use_model_defaults=False),trace='none',seed=None)
                finally:m.generate=original
                rows=[]
                for old,v in zip(saved,values):
                    assert list(v.token_ids)==old['token_ids'] and v.stop_reason==old['stop'];rows.append(dict(image_id=old['image_id'],token_ids=list(v.token_ids),text=tok.decode(v.token_ids,skip_special_tokens=False,clean_up_tokenization_spaces=False),stop=v.stop_reason))
                _write(out/'raw.json',dict(rows=rows));torch.save(captured,out/'capture.pt');rec['all_four_saved_sequences_exact']=True
            else:
                for window in p['windows']:
                    tensor={};history=window['history']
                    for name,candidate in window['candidates'].items():
                        tokens=candidate['tokens'];actions=history+tokens;histories=[]
                        for j,prompt in enumerate(batch.prompt_token_ids):
                            suffix=actions if j==target else saved[j]['token_ids'][:len(actions)]
                            suffix=suffix+[tok.pad_token_id]*(len(actions)-len(suffix));histories.append(list(prompt)+suffix)
                        inputs=exact_history_inputs(m,batch.inputs,histories,pad_token_id=tok.pad_token_id,logits_to_keep=len(tokens)+1)
                        z=m(**inputs).logits[target].float().detach().cpu();assert len(z)==len(tokens)+1
                        tensor[name]=dict(logits=z,head_input=last['head_input'].clone(),positions=last['positions'].clone(),token_ids=tokens,history=history,conditioning=window['policy_history'])
                    torch.save(tensor,out/(window['name']+'.pt'))
        for h in hooks:h.remove()
        assert versions=={n:v._version for n,v in m.named_parameters()};assert torch.equal(W,m.get_input_embeddings()(coords).detach())
        torch.cuda.synchronize();rec.update(status='candidate_complete',parameters_unchanged=True,elapsed_seconds=time.monotonic()-start,peak_reserved_bytes=torch.cuda.max_memory_reserved())
    except BaseException as e:
        rec.update(status='technical_invalid',error=repr(e),elapsed_seconds=time.monotonic()-start);_write(out/'receipt.json',rec);raise
    _write(out/'receipt.json',rec)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--mode',choices=['native','scores'],required=True);a=ap.parse_args();execute(a.root,a.mode)
