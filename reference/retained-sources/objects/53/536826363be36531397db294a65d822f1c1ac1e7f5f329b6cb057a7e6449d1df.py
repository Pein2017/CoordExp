"""Fixed R/H/W completion; minimal derivative of sealed history_rereading.py."""
import argparse
import copy
import inspect
import json
import os
import time
from pathlib import Path

import torch
from transformers import LogitsProcessor, LogitsProcessorList
from probes.training_set_completion import repetition_history_runtime as native
from probes.training_set_completion import successful_row_state as state

P = Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-successful-row-mechanism')
LAYERS = tuple(range(14, 21))
OFFSETS = (63, 67, 68, 76, 103)
TARGET = 1


def validate_source(donor, panel, identity, width):
    assert donor['input_identity'] == identity and donor['width'] == width
    prefix = panel['cases'][0]['target_prefix_token_ids']
    assert donor['prefix'][:54] == prefix[:54]
    assert donor['action_range'] == [54, 63]
    assert donor['cache_range'] == [width+54, width+63]
    assert donor['prefix'][54:63] == donor['row_tokens']
    assert donor['row_tokens'][0:4] == [151646, 22592, 151647, 151648]
    assert donor['row_tokens'][-1] == 151649
    assert donor['layer_ids'] == list(LAYERS)


def validate_wrapper(donor, width):
    observed={x['offset']:x for x in donor['trace']}
    for off,token in [(64,151646),(65,22592),(66,151647),(67,151648)]:
        assert observed[off]['ids'][TARGET][0]==token
        assert observed[off]['cache_position']==[width+off-1]
    for layer in LAYERS:
        for kind in ['wrapper_key','wrapper_value']:
            assert donor['tensors']['history'][str(layer)][kind].shape[-2]==3


def cpu_check(output):
    """Source token spans, not implementation-only index arithmetic."""
    panel = json.loads((P/'stage2/panels/F.json').read_text())
    prefix = panel['cases'][0]['target_prefix_token_ids']
    raw = json.loads((P/'stage2/runtime/native-F/raw.json').read_text())['rows'][TARGET]['token_ids']
    assert raw[:63] == prefix and raw[63:67] == [151646,22592,151647,151648]
    ident = json.loads((P/'stage2/runtime/native-F/receipt.json').read_text())['input_identity']
    width = len(ident['prompt_token_ids'][TARGET])
    # Padded prompt IDs from actual saved batch define physical positions.
    physical = ident['prompt_token_ids'][TARGET] + raw[:67]
    assert physical[width+54:width+63] == prefix[54:63]
    assert physical[width+55:width+64] != prefix[54:63]
    donor = dict(input_identity=ident, width=width, prefix=prefix, action_range=[54,63],
                 cache_range=[width+54,width+63],row_tokens=prefix[54:63],layer_ids=list(LAYERS))
    validate_source(donor,panel,ident,width)
    rejected=[]
    for name,change in [('offset',{'cache_range':[width+55,width+64]}),
                        ('source',{'row_tokens':raw[45:54]})]:
        bad={**donor,**change}
        try:validate_source(bad,panel,ident,width)
        except AssertionError:rejected.append(name)
    assert rejected==['offset','source']
    previous=P.parent/'2026-09-17-history-rereading-mechanism'
    real=torch.load(previous/'runtime/extract-S/capture.pt',map_location='cpu',weights_only=True)
    validate_wrapper(real,width)
    for name,offset,key,value in [('wrapper_role',66,'ids',[[151647],[151648],[151647],[151647]]),
                                  ('wrapper_position',66,'cache_position',[width+66])]:
        bad=copy.deepcopy(real)
        next(x for x in bad['trace'] if x['offset']==offset)[key]=value
        try:validate_wrapper(bad,width)
        except AssertionError:rejected.append(name)
    assert rejected==['offset','source','wrapper_role','wrapper_position']
    assert physical[width+63:width+66]==[151646,22592,151647]
    assert physical[width+66]==151648
    native._write(output,dict(status='passed',rejected=rejected,width=width,
        row_tokens=prefix[54:63],wrapper_tokens=raw[63:67],source=native._binding(P/'stage2/runtime/native-F/raw.json')))


def run(panel_path, output, mode, donor_path=None, residual=False, cache_patch=False, rebuild=None, wrapper_patch=False, wrapper_donor_path=None):
    panel=json.loads(panel_path.read_text()); native.fresh._check_sources(panel)
    case=panel['cases'][0]; assert case['target_position']==TARGET
    prefix=list(case['target_prefix_token_ids']);assert len(prefix)==63
    output.mkdir(parents=True,exist_ok=False)
    from transformers.models.qwen3_vl import modeling_qwen3_vl
    from transformers import cache_utils
    receipt=dict(status='running',mode=mode,residual=residual,cache_patch=cache_patch,wrapper_patch=wrapper_patch,
        pid=os.getpid(),panel=native._binding(panel_path),producer=native._binding(Path(__file__)),
        installed_sources=[native._binding(Path(inspect.getfile(x))) for x in [modeling_qwen3_vl,cache_utils]],
        intervention=dict(action_offset=67,history_actions=[54,63],wrapper_actions=[63,66],layers=list(LAYERS),target=TARGET,
                          history_restored_after_each_attention=True,current_KV_not_restored=True))
    native._write(output/'receipt.json',receipt)
    started=time.monotonic(); count=0; width=None; handles=[]; active={}; trace=[]
    tensors=dict(history={},restore={},residual={},head={},logits={},positions={},cache={})
    try:
        infer=native.fresh.InferConfig.model_validate(panel['config'])
        assert infer.model.dtype=='fp32' and infer.backend.hf.attn_implementation=='sdpa'
        qwen,identity=native.fresh.load_policy(infer,device=torch.device('cuda:0'))
        model=qwen.model.eval();language=model.model.language_model;assert len(language.layers)==28
        versions={n:p._version for n,p in model.named_parameters()}
        config=copy.deepcopy(panel['config']);config['data']['input_jsonl']=case['group']['input_jsonl']
        requests,_=native.fresh.build_bound_native_requests(qwen,config,case['group']['cases'])
        batch=native.fresh.prepare_native_inputs(qwen.processor,requests,device='cuda:0',record_media_identity=True)
        input_identity=native.fresh._input_identity(batch);width=batch.inputs['input_ids'].shape[1]
        saved_receipt=json.loads(Path(case['saved_receipt']['path']).read_text())
        assert input_identity==saved_receipt['input_identity']
        assert width==len(input_identity['prompt_token_ids'][TARGET])
        donor=torch.load(donor_path,map_location='cpu',weights_only=True) if donor_path else None
        if donor is not None:
            validate_source(donor,panel,input_identity,width)
            receipt['donor']=native._binding(donor_path)
        wrapper_donor=torch.load(wrapper_donor_path,map_location='cpu',weights_only=True) if wrapper_donor_path else donor
        if wrapper_patch:
            assert wrapper_donor is not None
            validate_source(wrapper_donor,panel,input_identity,width)
            validate_wrapper(wrapper_donor,width)
            receipt['wrapper_donor']=native._binding(wrapper_donor_path or donor_path)
        assert not (residual or cache_patch) or donor is not None
        # Residual donor may remain S while conditional cache donor is failed F2.
        residual_path=panel.get('residual_donor')
        residual_donor=torch.load(residual_path,map_location='cpu',weights_only=True) if residual_path else donor
        if residual_path:receipt['residual_donor']=native._binding(Path(residual_path))
        if residual:validate_source(residual_donor,panel,input_identity,width)
        forced=list(prefix)
        if rebuild:
            src=json.loads(rebuild.read_text())['rows'][TARGET]['token_ids']
            assert src[:63]==prefix and 151645 not in src[:68]
            forced=src[:68];receipt['rebuild_from']=native._binding(rebuild)
        limit=68 if mode=='extract' else 3084

        def counter(module,args,kwargs):
            nonlocal count
            count+=1;assert count<=limit
            off=count-1
            if off in range(54,69):
                ids=kwargs['input_ids'].detach().cpu()
                cp=kwargs['cache_position'].detach().cpu()
                trace.append(dict(offset=off,ids=ids.tolist(),cache_position=cp.tolist()))
                assert ids.shape==(4,1) and cp.tolist()==[width+off-1]
                if off==67:assert int(ids[TARGET,0])==151648

        def position_hook(module,args,kwargs):
            off=count-1
            if off in range(54,69) or off in OFFSETS:
                tensors['positions'][off]=kwargs['position_ids'].detach().cpu().clone()

        def attention_pre(layer):
            def hook(module,args,kwargs):
                if count-1!=67:return
                cache=kwargs['past_key_values'];entry=cache.layers[layer]
                assert entry.keys.shape[-2]==width+66
                lo,hi=width+54,width+63
                before_k=entry.keys[:,:,lo:hi,:].detach().clone()
                before_v=entry.values[:,:,lo:hi,:].detach().clone()
                # Capture later common wrapper separately; not part of the intervention.
                tensors['history'][str(layer)]=dict(key=before_k.cpu(),value=before_v.cpu(),
                    wrapper_key=entry.keys[:,:,hi:width+66,:].detach().cpu().clone(),
                    wrapper_value=entry.values[:,:,hi:width+66,:].detach().cpu().clone())
                before_all=(entry.keys.detach().clone(),entry.values.detach().clone())
                if cache_patch:
                    dk=donor['tensors']['history'][str(layer)]['key'][TARGET].to(entry.keys)
                    dv=donor['tensors']['history'][str(layer)]['value'][TARGET].to(entry.values)
                    assert dk.shape==entry.keys[TARGET,:,lo:hi,:].shape
                    entry.keys[TARGET,:,lo:hi,:]=dk;entry.values[TARGET,:,lo:hi,:]=dv
                if wrapper_patch:
                    wk=wrapper_donor['tensors']['history'][str(layer)]['wrapper_key'][TARGET].to(entry.keys)
                    wv=wrapper_donor['tensors']['history'][str(layer)]['wrapper_value'][TARGET].to(entry.values)
                    assert wk.shape==entry.keys[TARGET,:,hi:width+66,:].shape
                    entry.keys[TARGET,:,hi:width+66,:]=wk;entry.values[TARGET,:,hi:width+66,:]=wv
                tensors['restore'][str(layer)]=dict(
                    applied_key=entry.keys[:,:,lo:hi,:].detach().cpu().clone(),
                    applied_value=entry.values[:,:,lo:hi,:].detach().cpu().clone(),
                    applied_wrapper_key=entry.keys[:,:,hi:width+66,:].detach().cpu().clone(),
                    applied_wrapper_value=entry.values[:,:,hi:width+66,:].detach().cpu().clone())
                active[layer]=(cache,before_all,before_k,before_v)
            return hook

        def attention_post(layer):
            def hook(module,args,result):
                if count-1!=67:return result
                cache,before_all,bk,bv=active.pop(layer);entry=cache.layers[layer]
                lo,hi=width+54,width+63
                assert entry.keys.shape[-2]==width+67
                if cache_patch:
                    entry.keys[:,:,lo:hi,:]=bk;entry.values[:,:,lo:hi,:]=bv
                if wrapper_patch:
                    entry.keys[:,:,hi:width+66,:]=before_all[0][:,:,hi:width+66,:]
                    entry.values[:,:,hi:width+66,:]=before_all[1][:,:,hi:width+66,:]
                assert torch.equal(entry.keys[:,:,:-1,:],before_all[0])
                assert torch.equal(entry.values[:,:,:-1,:],before_all[1])
                tensors['restore'][str(layer)].update(restored_key=entry.keys[:,:,lo:hi,:].detach().cpu().clone(),
                    restored_value=entry.values[:,:,lo:hi,:].detach().cpu().clone(),
                    restored_wrapper_key=entry.keys[:,:,hi:width+66,:].detach().cpu().clone(),
                    restored_wrapper_value=entry.values[:,:,hi:width+66,:].detach().cpu().clone(),all_history_restored=True)
                return result
            return hook

        def residual_hook(layer):
            def hook(module,args,value):
                off=count-1
                if off not in OFFSETS:return value
                tensors['residual'].setdefault(off,{})[str(layer)]=dict(before=value[:,-1].detach().cpu().clone())
                if off==67 and layer==13 and residual:
                    value=state._replace_last_target(value,TARGET,residual_donor['tensors']['residual'][67]['13']['before'][TARGET])
                tensors['residual'][off][str(layer)]['after']=value[:,-1].detach().cpu().clone()
                return value
            return hook

        def head_hook(module,args):
            if count-1 in OFFSETS:tensors['head'][count-1]=args[0][:,-1].detach().cpu().clone()

        def output_hook(module,args,result):
            off=count-1
            if off in OFFSETS:tensors['logits'][off]=result.logits[:,-1].detach().cpu().clone()
            if off in [67,68]:
                _,v=state._cache_snapshot(result.past_key_values,offset=off,target_position=TARGET)
                tensors['cache'][off]=v

        handles=[model.register_forward_pre_hook(counter,with_kwargs=True),
                 language.register_forward_pre_hook(position_hook,with_kwargs=True),
                 model.get_output_embeddings().register_forward_pre_hook(head_hook),model.register_forward_hook(output_hook)]
        for l in LAYERS:
            handles.extend([language.layers[l].self_attn.register_forward_pre_hook(attention_pre(l),with_kwargs=True),
                            language.layers[l].self_attn.register_forward_hook(attention_post(l))])
        for l in [13,14,17,20]:handles.append(language.layers[l].register_forward_hook(residual_hook(l)))
        # Unchanged production native generation; only the same target forcing seam differs by prefix length.
        class Force(LogitsProcessor):
            def __call__(self,ids,scores):
                off=ids.shape[1]-width;assert off==count-1
                if off<54:assert int(scores[TARGET].argmax())==forced[off]
                return native._force(scores,TARGET,forced[off]) if off<len(forced) else scores
        original=model.generate
        def generate(**kwargs):
            assert kwargs['max_new_tokens']==limit and not kwargs.get('do_sample')
            return original(**kwargs,logits_processor=LogitsProcessorList([Force()]))
        model.generate=generate
        try:
            with torch.no_grad():
                values=native.fresh.generate_continuations(model,batch,extensions=[[]]*4,budgets=[limit]*4,
                    eos_token_id=151645,pad_token_id=qwen.tokenizer.pad_token_id,
                    policy=native.fresh.NativeGenerationPolicy(temperature=0,top_p=1,top_k=0,
                        repetition_penalty=1,use_model_defaults=False),trace='none',seed=None)
        finally:model.generate=original
        assert not active and len(tensors['history'])==7
        assert versions=={n:p._version for n,p in model.named_parameters()}
        rows=[]
        for i,(v,item) in enumerate(zip(values,case['group']['cases'])):
            ids=list(v.token_ids)
            rows.append(dict(image_id=native._image_id(item),token_ids=ids,text=qwen.tokenizer.decode(ids,skip_special_tokens=False),stop=v.stop_reason))
        native._write(output/'raw.json',dict(rows=rows))
        expected=json.loads(Path(case['source_sampled_raw']['path']).read_text())['rows']
        for i,(row,old) in enumerate(zip(rows,expected)):
            if mode=='extract':assert row['token_ids']==old['token_ids'][:68]
            elif i!=TARGET:assert row['token_ids']==old['token_ids'] and row['stop']==old['stop']
        assert rows[TARGET]['token_ids'][:len(forced)]==forced
        payload=dict(tensors=tensors,input_identity=input_identity,width=width,prefix=prefix,
            action_range=[54,63],cache_range=[width+54,width+63],row_tokens=prefix[54:63],
            layer_ids=list(LAYERS),trace=trace)
        torch.save(payload,output/'capture.pt')
        logits=tensors['logits'][67][TARGET];top=torch.topk(logits,2)
        competitors={str(i):dict(logit=float(logits[i]),rank=int((logits>logits[i]).sum())+1) for i in [151801,151670,151645]}
        receipt.update(status='candidate_complete',model_forwards=count,loaded_identity=identity,
            input_identity=input_identity,width=width,raw=native._binding(output/'raw.json'),capture=native._binding(output/'capture.pt'),
            fork=dict(argmax=int(top.indices[0]),top2_margin=float(top.values[0]-top.values[1]),
                      coord131_minus_coord0=float(logits[151801]-logits[151670]),competitors=competitors),
            history_restore_exact=True,parameter_versions_unchanged=True,companions_exact=True,
            elapsed_seconds=time.monotonic()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated())
    except BaseException as e:
        receipt.update(status='technical_invalid',error=repr(e),model_forwards=count,elapsed_seconds=time.monotonic()-started)
        raise
    finally:
        native._write(output/'receipt.json',receipt)
        for h in handles:h.remove()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--panel',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=['extract','full'],default='full');p.add_argument('--donor',type=Path)
    p.add_argument('--residual',action='store_true');p.add_argument('--cache-patch',action='store_true')
    p.add_argument('--rebuild',type=Path);p.add_argument('--cpu-check',action='store_true')
    p.add_argument('--wrapper-patch',action='store_true');p.add_argument('--wrapper-donor',type=Path);a=p.parse_args()
    if a.cpu_check:cpu_check(a.output)
    else:run(a.panel,a.output,a.mode,a.donor,a.residual,a.cache_patch,a.rebuild,a.wrapper_patch,a.wrapper_donor)
