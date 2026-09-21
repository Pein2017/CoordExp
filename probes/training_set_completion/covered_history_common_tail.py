"""Fixed common-tail conditional row scores; bounded native-cache qualification."""
from __future__ import annotations
import argparse,json,os,time
from pathlib import Path
import torch
from transformers import LogitsProcessor,LogitsProcessorList
from probes.dora_owner_learning.runtime import load_policy
from probes.training_set_completion.artifacts import literal_binding as _binding, write_pretty_json as _write
from src.qwen.input_identity import input_identity as _input_identity, tensor_hash as _tensor_hash
from src.config.inference import InferConfig
from src.inference.bound_requests import build_bound_native_requests
from src.qwen.native import prepare_native_inputs,exact_history_inputs
from src.qwen.generation import generate_continuations,NativeGenerationPolicy
from src.qwen.special_token_embeddings import SelectedDeltaOutputHead


def execute(root:Path,condition:str):
    start=time.monotonic();p=json.loads((root/'panel.json').read_text());w=next(w for w in p['windows'] if w['name']==condition);g=p['group'];target=p['target']
    out=root/'runtime'/condition;out.mkdir(parents=True,exist_ok=False)
    rec=dict(status='running',pid=os.getpid(),condition=condition,panel=_binding(root/'panel.json'),producer=_binding(Path(__file__)),model_forwards=0,vision_forwards=0,cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    _write(out/'receipt.json',rec)
    try:
        for b in p['sources']:assert _binding(Path(b['path']))==b,b['path']
        q,identity=load_policy(InferConfig.model_validate(p['config']),device=torch.device('cuda:0'));m=q.model.eval();tok=q.tokenizer
        requests,_=build_bound_native_requests(q,p['config'],g['cases']);batch=prepare_native_inputs(q.processor,requests,device='cuda:0',record_media_identity=True)
        accepted=json.loads(Path(p['accepted_onset_native']['path']).read_text());assert identity==accepted['loaded_identity'];assert _input_identity(batch)==accepted['input_identity']
        rec.update(loaded_identity=identity,input_identity=_input_identity(batch));width=batch.inputs['input_ids'].shape[1]
        head=m.get_output_embeddings();assert isinstance(head,SelectedDeltaOutputHead) and head.bias is None
        coords=torch.tensor(p['coordinate_ids'],device='cuda:0');assert coords.tolist()==[tok.convert_tokens_to_ids(f'<|coord_{i}|>') for i in range(1000)]
        lookup={int(t):i for i,t in enumerate(head.selected_token_ids.tolist())};ix=torch.tensor([lookup[int(t)] for t in coords],device='cuda:0')
        W=head.base.weight[coords].detach()+head.shared_embed_delta[ix].detach();E=m.get_input_embeddings()(coords).detach();assert torch.equal(W,E)
        cf=torch.load(p['coefficients']['path'],map_location='cpu',weights_only=True);assert _tensor_hash(W)==cf['effective_rows_sha256'];assert torch.equal(W.cpu().double().norm(dim=1),cf['norms'])
        torch.save(dict(output_rows=W.cpu(),input_rows=E.cpu(),coordinate_ids=coords.cpu(),norms=cf['norms'],factors=cf['factors'],bias=None),out/'readout.pt')
        rec['readout']=dict(effective_sha256=_tensor_hash(W),bias_exists=False,tied_base=head.base.weight.data_ptr()==m.get_input_embeddings().base.weight.data_ptr(),shared_delta=head.shared_embed_delta.data_ptr()==m.get_input_embeddings().shared_embed_delta.data_ptr())
        versions={n:v._version for n,v in m.named_parameters()};last={};stage=['scores'];active=[True]
        def count(module,args,kwargs):
            rec['model_forwards']+=1;assert rec['model_forwards']<=44
            if stage[0]=='incremental':active[0]=rec['model_forwards']-5>=29
        def vision(*args):rec['vision_forwards']+=1
        def pos(module,args,kwargs):
            if active[0]:last['positions']=kwargs['position_ids'][:,target].detach().cpu().clone()
        def capture(module,args):
            if active[0]:last['head_input']=args[0][target].detach().cpu().clone()
        hooks=[m.register_forward_pre_hook(count,with_kwargs=True),m.model.visual.register_forward_pre_hook(vision),m.model.language_model.register_forward_pre_hook(pos,with_kwargs=True),head.register_forward_pre_hook(capture)]
        saved=json.loads(Path(p['native_raw']['path']).read_text())['rows'];tensors={}
        with torch.inference_mode():
            for name,c in w['candidates'].items():
                tokens=c['tokens'];assert len(tokens)==10 and tok.encode(tok.decode(tokens,skip_special_tokens=False,clean_up_tokenization_spaces=False),add_special_tokens=False)==tokens
                actions=w['history']+tokens;histories=[]
                for j,prompt in enumerate(batch.prompt_token_ids):
                    suffix=actions if j==target else saved[j]['token_ids'][:len(actions)];suffix=suffix+[tok.pad_token_id]*(len(actions)-len(suffix));histories.append(list(prompt)+suffix)
                inputs=exact_history_inputs(m,batch.inputs,histories,pad_token_id=tok.pad_token_id,logits_to_keep=11)
                z=m(**inputs).logits[target].float().detach().cpu();assert len(z)==11
                tensors[name]=dict(logits=z,head_input=last['head_input'].clone(),positions=last['positions'].clone(),history=w['history'],token_ids=tokens,decoded_row=tok.decode(tokens,skip_special_tokens=False,clean_up_tokenization_spaces=False))
            torch.save(tensors,out/'scores.pt');stage[0]='incremental';forced=w['history']+w['candidates']['A1']['tokens'];assert len(forced)==39
            native={};seen=[]
            class CaptureComplete(Exception):pass
            class Supply(LogitsProcessor):
                def __call__(self,ids,scores):
                    off=ids.shape[1]-width;assert off==len(seen);seen.append(off)
                    assert ids[target,width:].tolist()==forced[:off]
                    for j,row in enumerate(saved):
                        if j==target:continue
                        expected=row['token_ids'][:off];expected+=[tok.pad_token_id]*(off-len(expected));assert ids[j,width:].tolist()==expected
                        if off<len(row['token_ids']):assert int(scores[j].argmax())==row['token_ids'][off],(j,off)
                    if off>=29:native[str(off)]=dict(logits=scores[target].detach().cpu().clone(),head_input=last['head_input'][-1].clone(),positions=last['positions'].clone(),history=ids[target,width:].tolist())
                    if off==39:
                        rec['last_all_batch_action_tokens']=ids[:,width:].tolist();raise CaptureComplete()
                    output=scores.clone();output[target].fill_(-torch.inf);output[target,forced[off]]=0;return output
            original=m.generate
            def wrap(**kwargs):
                assert kwargs['max_new_tokens']==3084 and kwargs['repetition_penalty']==1 and not kwargs['do_sample'];assert 'logits_processor' not in kwargs
                return original(**kwargs,logits_processor=LogitsProcessorList([Supply()]))
            m.generate=wrap
            try:
                try:generate_continuations(m,batch,extensions=[[]]*4,budgets=[3084]*4,eos_token_id=151645,pad_token_id=tok.pad_token_id,policy=NativeGenerationPolicy(temperature=0,top_p=1,top_k=0,repetition_penalty=1,use_model_defaults=False),trace='none',seed=None)
                except CaptureComplete:pass
                else:raise AssertionError('qualification did not reach capture boundary')
            finally:m.generate=original
            assert seen==list(range(40));torch.save(native,out/'incremental.pt')
            checks=[];a=tensors['A1']
            for j in range(11):
                n=native[str(29+j)];z=a['logits'][j];v=n['logits'];err=float((v-z).abs().max());margin=float(v.topk(2).values.diff().abs()[0]);assert int(v.argmax())==int(z.argmax()) and 2*err<margin,(j,err,margin)
                assert torch.equal(a['positions'][:,-11+j],n['positions'][:,-1]);checks.append(dict(row_offset=j,max_abs=err,margin=margin,twice_error_over_margin=2*err/margin,head_input_max_abs=float((a['head_input'][j]-n['head_input']).abs().max()),argmax_exact=True,positions_exact=True))
            _write(out/'parity.json',dict(status='passed',checks=checks,native_steps=seen,no_free_token_emitted=True,companions_exact=True,forced_history_token_ids=forced))
        for h in hooks:h.remove()
        assert versions=={n:v._version for n,v in m.named_parameters()};assert torch.equal(W,m.get_input_embeddings()(coords).detach());torch.cuda.synchronize()
        rec.update(status='candidate_complete',parameters_unchanged=True,elapsed_seconds=time.monotonic()-start,peak_reserved_bytes=torch.cuda.max_memory_reserved(),score_calls=4,incremental_calls=40,parity=_binding(out/'parity.json'))
    except BaseException as exc:
        rec.update(status='technical_invalid',error=repr(exc),elapsed_seconds=time.monotonic()-start);_write(out/'receipt.json',rec);raise
    _write(out/'receipt.json',rec)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--condition',choices=['A1','A2','B1','B2'],required=True);a=ap.parse_args();execute(a.root,a.condition)
