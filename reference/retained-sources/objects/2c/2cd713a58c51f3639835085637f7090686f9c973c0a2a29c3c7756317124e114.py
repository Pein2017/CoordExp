"""Mechanical-only serial first update for the exact frozen dense48 objective."""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import resource
import signal
import time
import numpy as np
import torch
from safetensors.torch import save_file, load_file

from src.artifacts import load_canonical_json
from probes.dora_owner_learning.selective_preservation_dense import (
    OUTPUT as DENSE, WIDE, AUTONOMOUS, CONFIG, OPTIMIZER,
    rank_items, global_image_loss, optimizer_hash,
)
from probes.dora_owner_learning.route_access import publish
from probes.dora_owner_learning.candidate_opportunity import file_hash, require
from probes.dora_owner_learning.branch_bridge import summarize_logits

OUT=AUTONOMOUS/'dense48-parity-serial'


def ordered_items(packet):
    items={x['key']:x for rank in range(8) for x in rank_items(packet,rank)}
    keys=[c['case_id'] for c in packet['cases']]+[c['example_id'] for c in packet['support_cases']]
    require(len(items)==len(keys)==50 and len(set(keys))==50,'serial50 coverage')
    return [items[k] for k in keys]


def cpu_check():
    logits=torch.full((3,151646),-1000.)
    logits[:,:2]=math.log(.5)
    ref=torch.full_like(logits,-1000.);ref[:,0]=math.log(.8);ref[:,1]=math.log(.2)
    item=dict(kind='support',positions=[0,1,2],action_ids=[0,1,151645])
    loss,ce,kl=global_image_loss(logits,torch.tensor(item['action_ids']),item,ref)
    expected=(10/48)*(.8*math.log(1.6)+.2*math.log(.4))
    require(abs(float(loss)-expected)<1e-7 and ce is None,'serial globally normalized loss; no DDP compensation')
    try:global_image_loss(logits,torch.tensor(item['action_ids']),{**item,'positions':[0,1]},ref)
    except ValueError:pass
    else:raise ValueError('wrong EOS mask accepted')
    return dict(global_weight_fixture=float(loss),expected=expected,wrong_mask_rejected=True)


def prepare():
    require(not (OUT/'inputs.json').exists(),'occupied control inputs')
    packet=load_canonical_json(DENSE/'inputs.json')
    items=ordered_items(packet)
    references={}
    sources={str(DENSE/'inputs.json'):file_hash(DENSE/'inputs.json'),str(Path(__file__)):file_hash(__file__)}
    for rank in range(8):
        path=DENSE/'ranks'/f'rank{rank}'/'references.json';sources[str(path)]=file_hash(path)
        for card in load_canonical_json(path):
            require(card['key'] not in references,'duplicate cache identity')
            require(file_hash(card['path'])==card['sha256'],'reference cache changed')
            sources[card['path']]=card['sha256'];references[card['key']]=card
    require(set(references)=={i['key'] for i in items},'all50 references required')
    for item in items:
        ref=references[item['key']]
        array=np.load(ref['path'],mmap_mode='r',allow_pickle=False)
        require(list(array.shape)==ref['shape']==[len(item['positions']),152670] and array.dtype==np.float32,'cache mask/state alignment')
    require(sum(r['cache_bytes'] for r in references.values())==3678125640,'reference cache bytes')
    for p,h in packet['source_files'].items():require(file_hash(p)==h,'bound Source input changed')
    publish(OUT/'inputs.json',dict(schema='dense48_serial_control.inputs.v1',mechanical_only=True,dense_packet=packet,
        items=items,references=references,source_files={**packet['source_files'],**sources},
        order='Original entries368/7116, then sortednumeric48 support IDs; globally normalized sum, no8x/DDP/rank mean.',
        bounds=dict(model_loads=1,reused_reference_caches=50,new_reference_forwards=0,train_forwards=50,entry_scores=4,total_forwards=54,seconds=1200),
        cpu_check=cpu_check()))


def compare_vectors(image,values):
    result={}
    for label,root in [('failed_DDP_dense48',DENSE),('wide31',WIDE)]:
        old=np.load(root/'scores'/f'step-01-{image}.npy',allow_pickle=False)
        diff=values.astype(np.float64)-old.astype(np.float64)
        result[label]=dict(max_abs=float(abs(diff).max()),RMS=float(np.sqrt(np.mean(diff*diff))),
            abs_quantiles={str(q):float(np.quantile(abs(diff),q)) for q in [0,.25,.5,.75,.9,.95,.99,.999,1]},
            exceeds_old1e5=int((abs(diff)>1e-5).sum()))
    return result


def run():
    from src.config.fingerprint import sha256_json
    from src.config.inference import load_research_infer_config
    from src.data import load_raw_examples
    from src.inference.runtime import assemble_frontend
    from src.qwen.native import prepare_replay
    from src.adapters.dora import select_dora_parameters
    from probes.dora_owner_learning.runtime import load_policy
    from probes.dora_owner_learning.train import _materialize_group,_tensor_state_hash,_save_adapter_only
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='0','GPU0 only')
    require(not (OUT/'launch.json').exists(),'existing serial control invocation')
    p=load_canonical_json(OUT/'inputs.json');packet=p['dense_packet']
    for path,sha in p['source_files'].items():require(file_hash(path)==sha,f'source/cache changed: {path}')
    config=load_research_infer_config(CONFIG).config
    require(str(config.adapter.path)==packet['model']['current_adapter']['root'] and str(config.embedding_delta.path)==packet['model']['source_embedding']['root'] and
            str(config.model.base_model)==packet['model']['base_model_path'],'original Source composition')
    require(config.model.dtype=='fp32' and config.backend.hf.attn_implementation=='sdpa' and config.backend.hf.patch_embed_linearization=='enabled','frozen numerics')
    for ident in (packet['model']['current_adapter'],packet['model']['source_embedding']):
        for f in ident['files']:require(file_hash(Path(ident['root'])/f['relative_path'])==f['sha256'],'Source payload bytes')
    publish(OUT/'launch.json',dict(pid=os.getpid(),started=time.time(),mechanical_only=True,CUDA_VISIBLE_DEVICES='0'))
    started=time.monotonic();counters=dict(model_loads=0,model_forwards=0,train_forwards=0,entry_scores=0,updates=0,recomputed_reference_forwards=0)
    status,error='failed',None
    def expired(*_):raise TimeoutError('1200-second serial control bound')
    signal.signal(signal.SIGALRM,expired);signal.alarm(1200)
    try:
        qwen,identity=load_policy(config,device=torch.device('cuda:0'));counters['model_loads']=1
        model=qwen.model;model.eval()
        require(identity['effective_settings']['observed_model_dtype']['parameter_dtype_names']==['torch.float32'] and
                identity['effective_settings']['observed_attn_implementation']=='sdpa' and identity['model_identity']['adapter']['merged_adapters']==[], 'loaded identity')
        publish(OUT/'model.json',identity)
        def count(*_):
            counters['model_forwards']+=1
            require(counters['model_forwards']<=54,'54-forward bound')
        model.register_forward_pre_hook(count)
        for parameter in model.parameters():parameter.requires_grad_(False)
        named=select_dora_parameters(model,towers=('language',),adapter_name='default')
        require(len(named)==588 and sum(t.numel() for _,t in named)==18006016 and all('language_model' in n for n,_ in named),'selected surface')
        for _,parameter in named:parameter.requires_grad_(True)
        selected={id(t) for _,t in named};frozen=[(n,t) for n,t in model.named_parameters() if id(t) not in selected]
        frozen_before=_tensor_state_hash(frozen);versions=[(t,t._version) for _,t in frozen]
        initial_adapter_hash=_tensor_state_hash(named);initial=[t.detach().clone() for _,t in named]
        layout=[];offset=0
        for name,t in named:
            layout.append(dict(name=name,shape=list(t.shape),numel=t.numel(),offset=offset,end=offset+t.numel(),dtype=str(t.dtype)));offset+=t.numel()
        publish(OUT/'gradient_layout.json',layout)
        optimizer=torch.optim.AdamW([t for _,t in named],**OPTIMIZER);require(not optimizer.state,'fresh AdamW')
        frontend=assemble_frontend(config,generation_config_fingerprint=sha256_json(config.generation.model_dump(mode='json')))
        raw={str(r.example_id):r for r in load_raw_examples(config.data.input_jsonl)}
        work=[];refs={};refversions={}
        for item in p['items']:
            c=item['case'];prompt,inputs,grid=_materialize_group(qwen=qwen,frontend=frontend,config=config,raw=raw[c['example_id']],group=c['group'])
            work.append((item,prompt,{**inputs,'image_grid_thw':grid}))
            array=np.load(p['references'][item['key']]['path'],allow_pickle=False)
            ref=torch.from_numpy(array).to('cuda:0').detach()
            require(list(ref.shape)==p['references'][item['key']]['shape'] and not ref.requires_grad and ref.grad_fn is None,'reused reference identity')
            refs[item['key']]=ref;refversions[item['key']]=ref._version
        torch.cuda.reset_peak_memory_stats()
        def entry_scores(step):
            scores={};arrays={}
            for item,prompt,inputs in work[:2]:
                c=item['case'];counters['entry_scores']+=1
                with torch.inference_mode():
                    replay=prepare_replay(model,inputs,prompt_token_ids=prompt,continuation_token_ids=c['state_ids']+[c['target_token_id']])
                    logits=replay.aligned_logits(model(**replay.inputs).logits)
                    require(logits.shape[0]==c['action_index']+1 and int(replay.target_ids[-1])==c['target_token_id'],'entry alignment')
                    vals=logits[-1].detach().cpu().numpy().copy();del logits,replay
                path=OUT/'scores'/f"step-{step:02d}-{c['image_id']}.npy";path.parent.mkdir(exist_ok=True)
                with path.open('xb') as stream:np.save(stream,vals,allow_pickle=False)
                require(np.array_equal(vals,np.load(path,allow_pickle=False)),'score reload')
                score=summarize_logits(vals,c['entrance']);score['target_id']=c['target_token_id'];scores[c['case_id']]=score;arrays[c['image_id']]=vals
                if step==0:require(np.array_equal(vals,np.load(DENSE/'scores'/f"step-00-{c['image_id']}.npy",allow_pickle=False)),'cold Source initial vector differs')
            publish(OUT/'scores'/f'step-{step:02d}.json',scores)
            return scores,arrays
        initial_scores,_=entry_scores(0)
        optimizer.zero_grad(set_to_none=True);losses=[]
        for item,prompt,inputs in work:
            counters['train_forwards']+=1
            replay=prepare_replay(model,inputs,prompt_token_ids=prompt,continuation_token_ids=item['action_ids'])
            logits=replay.aligned_logits(model(**replay.inputs).logits)
            loss,ce,kl=global_image_loss(logits,replay.target_ids,item,refs[item['key']])
            require(bool(torch.isfinite(loss)) and abs(float(kl.detach()))<=1e-6,'each initial KL must be approximately zero')
            loss.backward()  # No WORLD multiplier; no DDP; no rank-local average.
            losses.append(dict(key=item['key'],kind=item['kind'],globally_normalized_loss=float(loss.detach()),CE=float(ce.detach()) if ce is not None else None,
                               KL=float(kl.detach()),preserved_states=len(item['positions'])))
            del loss,ce,kl,logits,replay
        require(len(losses)==50 and all(t.grad is not None and bool(torch.isfinite(t.grad).all()) for _,t in named),'all50 finite gradients')
        require(all(t.grad is None and not t.requires_grad for _,t in frozen),'frozen gradients')
        def save_gradient(name):
            vector=torch.cat([t.grad.detach().flatten() for _,t in named]).cpu().contiguous()
            path=OUT/name;save_file({'gradient':vector},str(path),metadata={'meaning':'selected parameters in gradient_layout.json order'})
            require(torch.equal(load_file(str(path))['gradient'],vector),'gradient snapshot reload')
            return dict(path=str(path),sha256=file_hash(path),l2_float64=float(vector.double().norm()),
                        max_abs=float(vector.abs().max()),nonzero=int(torch.count_nonzero(vector)),tensor_state_hash=_tensor_state_hash([(n,t.grad) for n,t in named]))
        raw_gradient=save_gradient('raw_global_gradient.safetensors')
        raw_norm=float(torch.nn.utils.clip_grad_norm_([t for _,t in named],1.,error_if_nonfinite=True,foreach=False))
        clipped_gradient=save_gradient('clipped_global_gradient.safetensors')
        require(clipped_gradient['l2_float64']<=1.000001 and raw_norm>0,'global clip')
        optimizer.step();counters['updates']=1
        require({int(s['step']) for s in optimizer.state.values()}=={1},'one AdamW step')
        optimizer_path=OUT/'adam_after_step1.pt';torch.save(optimizer.state_dict(),optimizer_path)
        loaded=torch.load(optimizer_path,map_location='cpu',weights_only=True)
        current=optimizer.state_dict()
        require(loaded['param_groups']==current['param_groups'],'optimizer hyperparameter snapshot')
        for key,state in current['state'].items():
            for field,value in state.items():require(torch.equal(loaded['state'][key][field],value.cpu()),'Adam state snapshot reload')
        adapter=_save_adapter_only(model,source_root=Path(config.adapter.path),output=OUT/'mechanical_adapter')
        adapter_hash=_tensor_state_hash(named);opt_hash=optimizer_hash(optimizer,named)
        movement=math.sqrt(sum(float((t.detach()-old).double().square().sum()) for (_,t),old in zip(named,initial)))
        require(adapter_hash!=initial_adapter_hash and all(t._version==v for t,v in versions) and _tensor_state_hash(frozen)==frozen_before,'adapter/frozen identity')
        require(all(not x.requires_grad and x.grad_fn is None and x._version==refversions[k] for k,x in refs.items()),'reference cache mutation')
        # This receipt and all state snapshots exist before any post-step score or comparison.
        state=dict(mechanical_only=True,raw_gradient=raw_gradient,clipped_gradient=clipped_gradient,raw_clip_norm=raw_norm,
                   gradient_layout_sha256=file_hash(OUT/'gradient_layout.json'),optimizer=dict(path=str(optimizer_path),sha256=file_hash(optimizer_path),tensor_state_hash=opt_hash),
                   adapter=adapter,adapter_tensor_hash=adapter_hash,initial_adapter_tensor_hash=initial_adapter_hash,
                   frozen_tensor_hash_before=frozen_before,frozen_tensor_hash_after=frozen_before,parameter_delta_l2=movement,
                   counters_before_post_scores=dict(counters),publication_time=time.time())
        publish(OUT/'state_saved_before_comparisons.json',state)
        post_scores,arrays=entry_scores(1)
        comparisons={image:compare_vectors(image,vals) for image,vals in arrays.items()}
        for c in packet['cases']:
            own=post_scores[c['case_id']]
            for label,root in [('failed_DDP_dense48',DENSE),('wide31',WIDE)]:
                old=load_canonical_json(root/'scores'/'step-01.json')[c['case_id']]
                comparisons[c['image_id']][label]['entry_delta']={k:own[k]-old[k] for k in ['probability','logprob','A_vs_best_other_margin','A_vs_B_margin','rank_min']}
                comparisons[c['image_id']][label]['top1_equal']=own['top1_id']==old['top1_id']
        require(counters['model_forwards']==54 and counters['train_forwards']==50 and counters['entry_scores']==4,'exact control count')
        report=dict(schema='dense48_serial_parity.mechanical.v1',status='mechanical_control_completed',not_for_natural_evaluation=True,
                    updates=1,inputs_sha256=file_hash(OUT/'inputs.json'),state_snapshot_sha256=file_hash(OUT/'state_saved_before_comparisons.json'),
                    state=state,initial_scores=initial_scores,post_scores=post_scores,comparisons=comparisons,image_losses=losses,
                    objective_reconstruction=sum(x['globally_normalized_loss'] for x in losses),gate_revision=False,
                    resources=dict(**counters,cumulative_model_seconds=time.monotonic()-started,
                                   peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
                                   peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))
        publish(OUT/'mechanical_receipt.json',report);status='completed'
    except BaseException as exc:
        error=f'{type(exc).__name__}: {exc}';raise
    finally:
        signal.alarm(0)
        publish(OUT/'terminal.json',dict(status=status,error=error,mechanical_only=True,counters=counters,cumulative_model_seconds=time.monotonic()-started))


def verify():
    r=load_canonical_json(OUT/'mechanical_receipt.json')
    require(r['not_for_natural_evaluation'] and r['updates']==1 and r['resources']['model_forwards']==54,'mechanical scope')
    for name in ['raw_gradient','clipped_gradient','optimizer']:
        ref=r['state'][name];require(file_hash(ref['path'])==ref['sha256'],'state artifact hash')
    require(load_canonical_json(OUT/'terminal.json')['status']=='completed','control failed')
    for image in ('368','7116'):
        values=np.load(OUT/'scores'/f'step-01-{image}.npy',allow_pickle=False)
        replay=compare_vectors(image,values)
        for label in replay:
            require(all(replay[label][k]==r['comparisons'][image][label][k] for k in replay[label]),'comparison replay changed')
    print(json.dumps(dict(status=r['status'],comparisons=r['comparisons'],raw_gradient_norm=r['state']['raw_gradient']['l2_float64'],parameter_delta_l2=r['state']['parameter_delta_l2'],resources=r['resources'])))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','run','verify']);args=parser.parse_args()
    if args.command=='prepare':prepare();print(json.dumps(cpu_check()))
    elif args.command=='run':run()
    else:verify()
