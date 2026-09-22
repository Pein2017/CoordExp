"""Fixed dense48 Source-preservation arm using the existing eight-rank DDP pattern."""
from __future__ import annotations

import argparse
from contextlib import nullcontext, redirect_stdout, redirect_stderr
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import time

import torch

from src.artifacts import load_canonical_json
from .candidate_opportunity import digest, file_hash, indexed, require
from .route_access import ROOT, CONFIG, checked_ids, publish
from .entrance_ce import OPTIMIZER
from .selective_preservation import selective_loss
from .selective_preservation_wide import OUTPUT as WIDE
from .branch_bridge import summarize_logits

AUTONOMOUS = ROOT / '2026-09-10-selective-owner-learning-autonomous'
OUTPUT = AUTONOMOUS / 'soft-preservation-dense48/training'
FULL_PREFLIGHT = AUTONOMOUS / 'full-support-preflight.json'
DENSE_IDS = ['25274','49327','64010','90862','101636','114340','152252','158044','203986',
             '422969','474979','511251','532132','540107','548337','568311','575627']
WORLD, STEPS, SUPPORT = 8, 23, 48
CEILINGS = [216,168,144,144,144,144,144,144]


def compensate_for_ddp(globally_normalized_loss):
    return WORLD * globally_normalized_loss


def global_image_loss(logits, targets, item, reference_logp):
    """Globally normalized contribution, before DDP's world-size compensation."""
    if item['kind'] == 'entrance':
        return selective_loss(logits, targets, item['case']['entrance'], item['action_ids'],
                item['suffix_start'], item['positions'], reference_logp)
    checked_ids(item['action_ids'], 'im_end')
    require(item['positions'] == list(range(len(item['action_ids']))) and targets.tolist() == item['action_ids'] and
            logits.shape == reference_logp.shape and logits.shape[0] == len(item['action_ids']), 'support full-state alignment')
    ref = reference_logp.detach()
    kl = (ref.exp() * (ref - torch.log_softmax(logits, -1))).sum(-1).mean()
    return (10 / SUPPORT) * kl, None, kl


def rank_items(packet, rank):
    require(0 <= rank < WORLD, 'rank range')
    items = []
    if rank < 2:
        case = packet['cases'][rank]
        t = packet['trajectories'][case['case_id']]
        items.append(dict(key=case['case_id'], kind='entrance', case=case, action_ids=t['action_ids'],
                          positions=t['preservation_positions'], suffix_start=t['suffix_start']))
    for case in packet['support_cases'][rank::WORLD]:
        items.append(dict(key=case['example_id'], kind='support', case=case, action_ids=case['action_ids'],
                          positions=list(range(len(case['action_ids'])))))
    require(sum(x['kind']=='support' for x in items) == 6 and len(items) == (7 if rank < 2 else 6), 'rank work coverage')
    return items


class DenseScorer(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs, prompt, item, reference_logp):
        from src.qwen.native import prepare_replay
        replay = prepare_replay(self.model, inputs, prompt_token_ids=prompt, continuation_token_ids=item['action_ids'])
        logits = replay.aligned_logits(self.model(**replay.inputs).logits)
        loss, ce, kl = global_image_loss(logits, replay.target_ids, item, reference_logp)
        # DDP averages, so compensate exactly once. No rank-local image average.
        return compensate_for_ddp(loss), (ce.detach() if ce is not None else None), kl.detach()


def prepare(output):
    require(not output.exists(), 'occupied training root')
    wide = load_canonical_json(WIDE / 'inputs.json')
    full = load_canonical_json(FULL_PREFLIGHT)
    cards = {c['image_id']: c for c in full['all254_cases']}
    base_ids = wide['support_image_ids']
    require(len(base_ids) == 31 and not set(base_ids)&set(DENSE_IDS), 'original31/dense17 identity')
    expected_dense = sorted((c['image_id'] for c in cards.values() if c['admitted'] and c['predictions']>=20 and c['image_id'] not in base_ids),key=int)
    require(expected_dense == sorted(DENSE_IDS,key=int), 'explicit Source-only density admission')
    groups = indexed(load_canonical_json(ROOT / '2026-09-10-fixed-witness-route-access/inputs.json')['plan']['population']['groups'], 'example_id')
    support = []
    for image_id in sorted(base_ids + DENSE_IDS,key=int):
        card = cards[image_id]
        ids = checked_ids(card['action_ids'],'im_end')
        group = groups[card['example_id']]
        require(card['admitted'] and card['parser_drops']==0 and digest(ids)==card['action_ids_sha256'] and
                digest(group['prompt_token_ids'])==card['prompt_ids_sha256'] and
                group['executed_media_sha256']==card['executed_media_sha256'] and
                file_hash(group['image_path'])==card['image_content_sha256'], 'admitted token/media identity')
        support.append(dict(image_id=image_id,example_id=card['example_id'],action_ids=ids,
                            prompt_token_ids=group['prompt_token_ids'],group=group))
    require(len(support)==48 and sum(len(c['action_ids']) for c in support)==5848 and
            max(len(c['action_ids'])+len(c['prompt_token_ids']) for c in support)==1627, 'dense support bounds')
    require([c['image_id'] for c in wide['cases']]==['368','7116'], 'entry rank ownership')
    sources={**wide['source_files'],**full['source_files'],str(WIDE/'inputs.json'):file_hash(WIDE/'inputs.json'),str(FULL_PREFLIGHT):file_hash(FULL_PREFLIGHT)}
    comparators={}
    for case in wide['cases']:
        path=WIDE/'scores'/f"step-01-{case['image_id']}.npy"
        sources[str(path)]=file_hash(path)
        comparators[case['case_id']]=str(path)
    for path,sha in sources.items():
        require(file_hash(path)==sha,f'frozen source changed: {path}')
    packet=dict(schema_version='selective_preservation_dense.inputs.v1',cases=wide['cases'],trajectories=wide['trajectories'],
        model=wide['model'],train_source=wide['train_source'],qualifications=wide['qualifications'],support_cases=support,
        support_image_ids=[c['image_id'] for c in support],support_action_states=5848,added_dense_ids=DENSE_IDS,
        support_preflight_sha256=file_hash(FULL_PREFLIGHT),source_files=sources,step1_comparators=comparators,
        optimizer=OPTIMIZER,lambda_kl=10.,lambda_support_kl=10.,updates=23,clip_gradient_norm=1.,
        objective='mean2CE+10mean2KL+10mean48KL; globally normalized per-image loss multiplied by8 before DDP average; one global clip after reduction.',
        rank_support_ids={str(r):[c['image_id'] for c in support[r::8]] for r in range(8)},
        bounds=dict(model_loads=8,reference_forwards=50,train_forwards=1150,entry_score_forwards=48,model_forwards=1248,
                    rank_model_forwards=CEILINGS,reference_cache_bytes=3678125640,rank_seconds=3600,distributed_timeout_seconds=600))
    for r in range(8): rank_items(packet,r)
    output.mkdir(parents=True,exist_ok=False)
    publish(output/'inputs.json',packet)
    files=[Path(__file__),Path(__file__).with_name('tests')/'test_selective_preservation_dense.py',CONFIG,
           Path(__file__).with_name('train.py'),Path(__file__).with_name('runtime.py'),Path(__file__).with_name('selective_preservation.py'),
           Path(__file__).with_name('selective_preservation_wide.py'),Path(__file__).with_name('entrance_ce.py'),Path(__file__).with_name('branch_bridge.py')]
    files+=list(Path('src/qwen').glob('*.py'))+[Path('src/losses/token_scores.py'),Path('src/adapters/dora.py')]
    saved=[]
    for path in files:
        target=output/'effective_code'/str(path.resolve()).lstrip('/')
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,target)
        saved.append(dict(path=str(path.resolve()),staged=str(target),sha256=file_hash(target)))
    publish(output/'code_identity.json',dict(files=saved,git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                                           git_status=subprocess.check_output(['git','status','--short'],text=True)))
    return packet


def optimizer_hash(optimizer,named):
    from .train import _tensor_state_hash
    return _tensor_state_hash([(f'{name}/{key}',optimizer.state[p][key]) for name,p in named for key in ('step','exp_avg','exp_avg_sq')])


def execute_rank(output):
    import numpy as np
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from src.config.fingerprint import sha256_json
    from src.config.inference import load_research_infer_config
    from src.data import load_raw_examples
    from src.inference.runtime import assemble_frontend
    from src.qwen.native import prepare_replay
    from src.adapters.dora import select_dora_parameters
    from .runtime import load_policy
    from .train import (_materialize_group,_parameter_layout,_tensor_state_hash,_save_adapter_only,
                        _dist_values,_all_true,EXPECTED_TRAINABLE_TENSORS,EXPECTED_TRAINABLE_SCALARS)
    rank,local,world=[int(os.environ.get(k,'-1')) for k in ('RANK','LOCAL_RANK','WORLD_SIZE')]
    require(world==8 and rank==local and 0<=rank<8 and os.environ.get('CUDA_VISIBLE_DEVICES')=='0,1,2,3,4,5,6,7','eight GPU topology')
    run=output/'ranks'/f'rank{rank}'
    run.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    counters=dict(model_loads=0,model_forwards=0,image_forwards=0,reference_forwards=0,train_forwards=0,entry_score_forwards=0,
                  updates=0,supervised_target_tokens=0,support_KL_trajectories=0,explicit_collective_calls=0,ddp_synchronized_backwards=0)
    status,error='failed',None
    final_state={}
    def expired(*_):raise TimeoutError('3600-second rank invocation ceiling')
    signal.signal(signal.SIGALRM,expired)
    signal.alarm(3600)
    with (run/'execution.log').open('x') as log,redirect_stdout(log),redirect_stderr(log):
      try:
        torch.cuda.set_device(local)
        device=torch.device('cuda',local)
        dist.init_process_group('nccl',timeout=timedelta(seconds=600),device_id=device)
        def gather(value):
            counters['explicit_collective_calls']+=1
            return _dist_values(value)
        def all_true(value):
            counters['explicit_collective_calls']+=1
            return _all_true(value,device)
        def barrier():
            counters['explicit_collective_calls']+=1
            dist.barrier()
        packet=load_canonical_json(output/'inputs.json')
        require(len(set(gather(file_hash(output/'inputs.json'))))==1,'rank input identity')
        for path,sha in packet['source_files'].items(): require(file_hash(path)==sha,f'frozen source changed: {path}')
        config=load_research_infer_config(CONFIG).config
        require(str(config.model.base_model)==packet['model']['base_model_path'] and str(config.adapter.path)==packet['model']['current_adapter']['root'] and
                str(config.embedding_delta.path)==packet['model']['source_embedding']['root'],'original Source composition')
        require(config.model.dtype=='fp32' and config.backend.hf.attn_implementation=='sdpa' and config.backend.hf.patch_embed_linearization=='enabled','frozen numerics')
        for ident in (packet['model']['current_adapter'],packet['model']['source_embedding']):
            for f in ident['files']:require(file_hash(Path(ident['root'])/f['relative_path'])==f['sha256'],'immutable Source files')
        qwen,model_identity=load_policy(config,device=device)
        counters['model_loads']=1
        model=qwen.model
        model.eval()
        require(model_identity['effective_settings']['observed_model_dtype']['parameter_dtype_names']==['torch.float32'] and
                model_identity['effective_settings']['observed_attn_implementation']=='sdpa' and
                model_identity['model_identity']['adapter']['merged_adapters']==[],'loaded numerical identity')
        publish(run/'model.json',model_identity)
        def count_model(*_):
            counters['model_forwards']+=1
            require(counters['model_forwards']<=CEILINGS[rank],'rank model forward ceiling')
        def count_image(*_):counters['image_forwards']+=1
        model.register_forward_pre_hook(count_model)
        visual=[m for n,m in model.named_modules() if n.endswith('visual')]
        require(len(visual)==1,'vision counter identity')
        visual[0].register_forward_pre_hook(count_image)
        for p in model.parameters():p.requires_grad_(False)
        named=select_dora_parameters(model,towers=('language',),adapter_name='default')
        require(len(named)==EXPECTED_TRAINABLE_TENSORS==588 and sum(p.numel() for _,p in named)==EXPECTED_TRAINABLE_SCALARS==18006016 and
                all('language_model' in n and not any(x in n for x in ('visual','merger','embed_tokens','lm_head')) for n,_ in named),'DoRA training surface')
        for _,p in named:p.requires_grad_(True)
        selected={id(p) for _,p in named}
        frozen=[(n,p) for n,p in model.named_parameters() if id(p) not in selected]
        initial_adapter=_tensor_state_hash(named)
        initial_frozen=_tensor_state_hash(frozen)
        require(len(set(gather((initial_adapter,initial_frozen))))==1,'initial rank tensors differ')
        scorer=DenseScorer(model)
        ddp=DDP(scorer,device_ids=[local],output_device=local,broadcast_buffers=False,init_sync=False)
        # Freeze/version snapshot follows DDP construction, as in the repaired shared choreography.
        frozen_hash=_tensor_state_hash(frozen)
        require(frozen_hash==initial_frozen,'DDP construction changed frozen bytes')
        versions=[(p,p._version) for _,p in frozen]
        original=[p.detach().clone() for _,p in named]
        layout=_parameter_layout(named)
        publish(run/'trainable_layout.json',layout)
        optimizer=torch.optim.AdamW([p for _,p in named],**OPTIMIZER)
        require(not optimizer.state,'fresh optimizer required')
        frontend=assemble_frontend(config,generation_config_fingerprint=sha256_json(config.generation.model_dump(mode='json')))
        raw={str(r.example_id):r for r in load_raw_examples(config.data.input_jsonl)}
        materialized=[]
        for item in rank_items(packet,rank):
            c=item['case']
            prompt,inputs,grid=_materialize_group(qwen=qwen,frontend=frontend,config=config,raw=raw[c['example_id']],group=c['group'])
            materialized.append((item,prompt,{**inputs,'image_grid_thw':grid}))
        score_inputs=[]
        if rank==0:
            for c in packet['cases']:
                prompt,inputs,grid=_materialize_group(qwen=qwen,frontend=frontend,config=config,raw=raw[c['example_id']],group=c['group'])
                score_inputs.append((c,prompt,{**inputs,'image_grid_thw':grid}))
        torch.cuda.reset_peak_memory_stats()
        refs,refversions,refcards={},{},[]
        for item,prompt,inputs in materialized:
            counters['reference_forwards']+=1
            with torch.no_grad():
                replay=prepare_replay(model,inputs,prompt_token_ids=prompt,continuation_token_ids=item['action_ids'])
                logits=replay.aligned_logits(model(**replay.inputs).logits)
                ref=torch.log_softmax(logits[item['positions']],-1).detach().clone()
                _,_,kl=global_image_loss(logits,replay.target_ids,item,ref)
                require(abs(float(kl))<=1e-6 and not ref.requires_grad and ref.grad_fn is None,'initial reference KL')
                refs[item['key']]=ref;refversions[item['key']]=ref._version
                values=ref.cpu().numpy().copy()
                del logits,replay
            path=run/'references'/f"{item['case']['image_id']}.npy";path.parent.mkdir(exist_ok=True)
            with path.open('xb') as stream:np.save(stream,values,allow_pickle=False)
            require(np.array_equal(values,np.load(path,allow_pickle=False)),'reference disk reload')
            refcards.append(dict(key=item['key'],kind=item['kind'],path=str(path),sha256=file_hash(path),shape=list(ref.shape),
                                 cache_bytes=ref.numel()*ref.element_size(),initial_KL=float(kl)))
        publish(run/'references.json',refcards)
        refstats=gather(dict(rank=rank,reference_count=len(refcards),cache_bytes=sum(c['cache_bytes'] for c in refcards)))
        require(sum(s['reference_count'] for s in refstats)==50 and sum(s['cache_bytes'] for s in refstats)==3678125640,'global reference bounds')
        def score_phase(step):
            barrier()
            result=None
            if rank==0:
              try:
                scores={};parity={}
                for c,prompt,inputs in score_inputs:
                    counters['entry_score_forwards']+=1
                    with torch.inference_mode():
                        replay=prepare_replay(model,inputs,prompt_token_ids=prompt,continuation_token_ids=c['state_ids']+[c['target_token_id']])
                        logits=replay.aligned_logits(model(**replay.inputs).logits)
                        require(logits.shape[0]==c['action_index']+1 and int(replay.target_ids[-1])==c['target_token_id'],'isolated causal alignment')
                        values=logits[-1].detach().cpu().numpy().copy();del logits,replay
                    path=output/'scores'/f"step-{step:02d}-{c['image_id']}.npy";path.parent.mkdir(exist_ok=True)
                    with path.open('xb') as stream:np.save(stream,values,allow_pickle=False)
                    require(np.array_equal(values,np.load(path,allow_pickle=False)),'score disk reload')
                    s=summarize_logits(values,c['entrance']);s['target_id']=c['target_token_id'];scores[c['case_id']]=s
                    if step==1:
                        old=np.load(packet['step1_comparators'][c['case_id']],allow_pickle=False)
                        parity[c['case_id']]=float(np.max(np.abs(values-old)))
                publish(output/'scores'/f'step-{step:02d}.json',scores)
                if step==1:
                    publish(output/'step1-vector-parity.json',dict(max_abs_errors=parity,tolerance=1e-5))
                    require(all(d<=1e-5 for d in parity.values()),'step1 full-vocabulary vector parity failed')
                result=dict(ok=True,scores=scores,step1_parity=parity)
              except BaseException as exc:result=dict(ok=False,error=f'{type(exc).__name__}: {exc}')
            result=gather(result)[0]
            require(result['ok'],f"rank0 score phase failed: {result.get('error')}")
            return result
        initial_scores=score_phase(0)['scores']
        dose=[];step1_parity={}
        for step in range(1,24):
            optimizer.zero_grad(set_to_none=True)
            before=[p.detach().clone() for _,p in named]
            local_losses=[];finite=True
            for i,(item,prompt,inputs) in enumerate(materialized):
                sync=i==len(materialized)-1
                with nullcontext() if sync else ddp.no_sync():
                    counters['train_forwards']+=1
                    loss,ce,kl=ddp(inputs,prompt,item,refs[item['key']])
                    finite=finite and bool(torch.isfinite(loss)) and float(kl)>=-1e-6
                    if step==1:finite=finite and abs(float(kl))<=1e-6
                    loss.backward()
                counters['supervised_target_tokens']+=int(item['kind']=='entrance')
                local_losses.append(dict(key=item['key'],kind=item['kind'],global_weighted_loss=float(loss.detach())/8,
                                         CE=float(ce) if ce is not None else None,mean_KL=float(kl),states=len(item['positions'])))
                del loss,ce,kl
            counters['ddp_synchronized_backwards']+=1
            finite=finite and all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for _,p in named)
            require(all_true(finite),'nonfinite/missing globally reduced gradients')
            require(all(p.grad is None and not p.requires_grad for _,p in frozen),'frozen gradients')
            gradient_hash=_tensor_state_hash([(n,p.grad) for n,p in named])
            require(len(set(gather(gradient_hash)))==1,'reduced gradient differs by rank')
            raw_norm=float(torch.nn.utils.clip_grad_norm_([p for _,p in named],1.,error_if_nonfinite=True,foreach=False))
            require(math.isfinite(raw_norm) and raw_norm>0,'invalid global gradient norm')
            clipped_norm=math.sqrt(sum(float(p.grad.double().square().sum()) for _,p in named))
            require(clipped_norm<=1.000001,'global clip bound')
            optimizer.step();counters['updates']=step
            require(all(p._version==v for p,v in versions),'frozen version changed')
            require(all(not x.requires_grad and x.grad_fn is None and x._version==refversions[k] for k,x in refs.items()),'reference cache changed')
            adapter_hash=_tensor_state_hash(named);opt_hash=optimizer_hash(optimizer,named)
            state_hashes=gather((adapter_hash,opt_hash))
            require(len(set(state_hashes))==1,'rank adapter/optimizer differs')
            require({int(s['step']) for s in optimizer.state.values()}=={step},'optimizer step identity')
            movement=math.sqrt(sum(float((p.detach()-old).double().square().sum()) for (_,p),old in zip(named,before)))
            total_movement=math.sqrt(sum(float((p.detach()-old).double().square().sum()) for (_,p),old in zip(named,original)))
            del before
            require(movement>0 and math.isfinite(total_movement),'invalid parameter movement')
            if step<=2:
                require(all_true(_tensor_state_hash(frozen)==frozen_hash),'smoke frozen bytes changed')
            perrank=gather(dict(rank=rank,losses=local_losses,gradient_hash=gradient_hash,adapter_hash=adapter_hash,optimizer_hash=opt_hash,
                              raw_gradient_norm=raw_norm,clipped_gradient_norm=clipped_norm,parameter_movement_l2=movement,
                              max_KL=max(x['mean_KL'] for x in local_losses),counters=dict(counters)))
            score_result=score_phase(step);final_scores=score_result['scores']
            if step==1:step1_parity=score_result['step1_parity']
            if step==2:require(any(s['max_KL']>1e-8 for s in perrank),'step2 nonzero KL not exercised')
            row=dict(update=step,rank=rank,local_losses=local_losses,gradient_hash=gradient_hash,adapter_hash=adapter_hash,optimizer_hash=opt_hash,
                     raw_gradient_norm=raw_norm,clipped_gradient_norm=clipped_norm,source_parameter_delta_l2=total_movement,
                     counters=dict(counters),seconds=time.monotonic()-started)
            publish(run/f'update-{step:02d}.json',row)
            if rank==0:
                dose.append(dict(update=step,ranks=perrank,final_scores=final_scores))
                publish(output/f'update-{step:02d}.json',dose[-1])
                if step==2:
                    publish(output/'two-step-smoke.json',dict(status='passed',updates=[1,2],initial_reference_stats=refstats,
                        step1_full_vocab_max_abs_errors=step1_parity,step2_max_KL_by_rank={str(s['rank']):s['max_KL'] for s in perrank},
                        reduced_gradient_adapter_optimizer_rank_identity=True,frozen_bytes_unchanged=True,source_parameter_delta_l2=total_movement))
                    print('TWO_STEP_SMOKE_COMPLETED',flush=True)
            # All ranks confirm rank0 has published the gate before beginning update3.
            if step==2:barrier()
            print(json.dumps(dict(rank=rank,update=step,seconds=time.monotonic()-started)),flush=True)
        require(all_true(_tensor_state_hash(frozen)==frozen_hash),'terminal frozen bytes changed')
        require(counters['model_forwards']==CEILINGS[rank] and counters['train_forwards']==23*len(materialized),'rank counters incomplete')
        final_state=dict(adapter_hash=adapter_hash,optimizer_hash=opt_hash,frozen_hash=frozen_hash,source_adapter_hash=initial_adapter,
                         reference_cache_bytes=sum(c['cache_bytes'] for c in refcards),references=refcards,
                         peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
                         peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,parameter_delta_l2=total_movement)
        ranks=gather(dict(rank=rank,counters=dict(counters),state=final_state))
        save_result=None
        if rank==0:
          try:
            adapter=_save_adapter_only(model,source_root=Path(config.adapter.path),output=output/'adapter')
            provisional=dict(schema_version='selective_preservation_dense.training.v1',status='unsealed',adapter=adapter,
                source_adapter=packet['model']['current_adapter'],source_embedding=packet['model']['source_embedding'],
                config=config.model_dump(mode='json'),cases=packet['cases'],final_scores=final_scores,initial_scores=initial_scores,
                updates=23,stop_reason='fixed_steps',optimizer=OPTIMIZER,lambda_kl=10.,lambda_support_kl=10.,clip_gradient_norm=1.,
                support_image_ids=packet['support_image_ids'],support_action_states=5848,support_preflight_sha256=packet['support_preflight_sha256'],
                objective=packet['objective'],trainable_layout=layout,dose=dose,rank_training_states=ranks,
                inputs_sha256=file_hash(output/'inputs.json'),code_identity_sha256=file_hash(output/'code_identity.json'),
                two_step_smoke_sha256=file_hash(output/'two-step-smoke.json'))
            publish(output/'provisional.json',provisional)
            save_result=dict(ok=True,adapter_fingerprint=adapter['fingerprint'])
          except BaseException as exc:save_result=dict(ok=False,error=f'{type(exc).__name__}: {exc}')
        save_result=gather(save_result)[0]
        require(save_result['ok'],f"rank0 save failed: {save_result.get('error')}")
        require(all_true(all(file_hash(Path(ident['root'])/f['relative_path'])==f['sha256'] for ident in (packet['model']['current_adapter'],packet['model']['source_embedding']) for f in ident['files'])),'Source disk bytes changed')
        status='completed'
      except BaseException as exc:
        error=f'{type(exc).__name__}: {exc}'
        raise
      finally:
        signal.alarm(0)
        publish(run/'terminal.json',dict(rank=rank,status=status,error=error,updates=counters['updates'],model_forwards=counters['model_forwards'],
            pid=os.getpid(),counters=counters,state=final_state,cumulative_model_seconds=time.monotonic()-started))
        if dist.is_initialized():dist.destroy_process_group()


def finalize(output):
    require(not (output/'receipt.json').exists(),'occupied sealed receipt')
    exit_path=output/'launcher_exit.json';exit_record=load_canonical_json(exit_path)
    require(exit_record['exit_code']==0,'launcher did not complete successfully')
    require({p.name for p in (output/'ranks').iterdir() if p.is_dir()}=={f'rank{r}' for r in range(8)},'exact rank directory coverage')
    terminals=[];records=[]
    for rank in range(8):
        path=output/'ranks'/f'rank{rank}'/'terminal.json';t=load_canonical_json(path)
        require(t['rank']==rank and t['status']=='completed' and t['updates']==23 and t['model_forwards']==CEILINGS[rank], 'rank terminal incomplete')
        terminals.append(t);records.append(dict(rank=rank,path=str(path),sha256=file_hash(path)))
    require(len({t['state']['adapter_hash'] for t in terminals})==1 and len({t['state']['optimizer_hash'] for t in terminals})==1,
            'terminal rank state disagreement')
    require(sum(t['model_forwards'] for t in terminals)==1248 and sum(t['counters']['supervised_target_tokens'] for t in terminals)==46,'global counters')
    provisional=load_canonical_json(output/'provisional.json')
    for ident in (provisional['adapter'],provisional['source_embedding']):
        for f in ident['files']:require(file_hash(Path(ident['root'])/f['relative_path'])==f['sha256'],'sealed checkpoint byte mismatch')
    receipt={**provisional,'status':'completed','rank_terminals':records,'launcher_exit':dict(path=str(exit_path),sha256=file_hash(exit_path)),
             'global_model_forwards':1248,'resources':dict(model_loads=8,reference_forwards=sum(t['counters']['reference_forwards'] for t in terminals),
             train_forwards=sum(t['counters']['train_forwards'] for t in terminals),entry_score_forwards=sum(t['counters']['entry_score_forwards'] for t in terminals),
             cumulative_rank_model_seconds=sum(t['cumulative_model_seconds'] for t in terminals),max_rank_model_seconds=max(t['cumulative_model_seconds'] for t in terminals),
             reference_cache_bytes=sum(t['state']['reference_cache_bytes'] for t in terminals),
             peak_cuda_allocated_bytes_max_rank=max(t['state']['peak_cuda_allocated_bytes'] for t in terminals),
             peak_cuda_reserved_bytes_max_rank=max(t['state']['peak_cuda_reserved_bytes'] for t in terminals),
             peak_rss_bytes_max_rank=max(t['state']['peak_rss_bytes'] for t in terminals)),
             'provisional_sha256':file_hash(output/'provisional.json')}
    publish(output/'receipt.json',receipt)
    return receipt


def launch(output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='0,1,2,3,4,5,6,7','launcher GPU topology')
    require(not (output/'launcher_owner.json').exists(),'existing launcher invocation')
    command=[sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc-per-node=8',
             '-m','probes.dora_owner_learning.selective_preservation_dense','rank','--output',str(output)]
    publish(output/'launcher_owner.json',dict(pid=os.getpid(),command=command,started=time.time(),
        processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],text=True)))
    started=time.monotonic()
    with (output/'torchrun.log').open('x') as stream:
        result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=False)
    publish(output/'launcher_exit.json',dict(exit_code=result.returncode,elapsed_seconds=time.monotonic()-started,finished=time.time()))
    if result.returncode:
        return result.returncode
    receipt=finalize(output)
    print(json.dumps(dict(status='completed',receipt_sha256=file_hash(output/'receipt.json'),resources=receipt['resources'])))
    return 0


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['prepare','rank','launch','verify'])
    parser.add_argument('--output',type=Path,default=OUTPUT)
    args=parser.parse_args()
    if args.command=='prepare':
        p=prepare(args.output);print(json.dumps(dict(support=len(p['support_cases']),bounds=p['bounds'])))
    elif args.command=='rank':execute_rank(args.output)
    elif args.command=='launch':raise SystemExit(launch(args.output))
    else:
        r=load_canonical_json(args.output/'receipt.json')
        require(r['status']=='completed' and r['global_model_forwards']==1248,'sealed receipt incomplete')
        for ref in r['rank_terminals']+[r['launcher_exit']]:require(file_hash(ref['path'])==ref['sha256'],'terminal/launcher binding changed')
        for ident in (r['adapter'],r['source_embedding']):
            for f in ident['files']:require(file_hash(Path(ident['root'])/f['relative_path'])==f['sha256'],'checkpoint file changed')
        print(json.dumps(dict(status=r['status'],updates=r['updates'],resources=r['resources'],final_margins={c:s['A_vs_best_other_margin'] for c,s in r['final_scores'].items()})))


if __name__=='__main__':main()
