"""Fixed support100 arm, derived from the accepted eight-rank dense48 path."""
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
from .candidate_opportunity import file_hash,require
from .route_access import ROOT,CONFIG,publish
from .entrance_ce import OPTIMIZER
from .branch_bridge import summarize_logits
from .selective_preservation_dense import (
    RETRY_OUTPUT as PRIOR,global_image_loss as coefficient10_loss,
    rank_items,optimizer_hash,compensate_for_ddp,CEILINGS,
)

OUTPUT=ROOT/'2026-09-10-selective-owner-learning-autonomous/soft-preservation-dense48-strong100/training'
NUMERICAL_ADMISSION='ddp_path_reuse_support100_v1'


def global_image_loss(logits,targets,item,reference_logp):
    loss,ce,kl=coefficient10_loss(logits,targets,item,reference_logp)
    return (loss*10 if item['kind']=='support' else loss),ce,kl


class StrongScorer(torch.nn.Module):
    def __init__(self,model):
        super().__init__();self.model=model
    def forward(self,inputs,prompt,item,reference_logp):
        from src.qwen.native import prepare_replay
        replay=prepare_replay(self.model,inputs,prompt_token_ids=prompt,continuation_token_ids=item['action_ids'])
        logits=replay.aligned_logits(self.model(**replay.inputs).logits)
        loss,ce,kl=global_image_loss(logits,replay.target_ids,item,reference_logp)
        return compensate_for_ddp(loss),ce.detach() if ce is not None else None,kl.detach()


def coefficient_fixture():
    import numpy as np
    vocab=151646
    logits=torch.full((3,vocab),-1000.,dtype=torch.float64)
    logits[:,:2]=math.log(.5);logits.requires_grad_(True)
    ref=torch.full_like(logits,-1000.);ref[:,0]=math.log(.8);ref[:,1]=math.log(.2)
    item=dict(kind='support',positions=[0,1,2],action_ids=[0,1,151645])
    targets=torch.tensor(item['action_ids'])
    old,_,_=coefficient10_loss(logits,targets,item,ref)
    new,_,_=global_image_loss(logits,targets,item,ref)
    a=torch.autograd.grad(old,logits,retain_graph=True)[0]
    b=torch.autograd.grad(new,logits)[0]
    error=float((b-10*a).abs().max())
    require(torch.equal(new,old*10) and error<1e-15,'support coefficient/gradient ratio10')
    entry=dict(kind='entrance',case={'entrance':dict(action_index=1,state_ids=[0],A_id=1)},
               action_ids=[0,1,2,151645],positions=[0,3],suffix_start=3)
    x=torch.zeros(4,vocab,dtype=torch.float32);p=torch.log_softmax(x[entry['positions']],-1)
    old_two=coefficient10_loss(x,torch.tensor(entry['action_ids']),entry,p)
    new_two=global_image_loss(x,torch.tensor(entry['action_ids']),entry,p)
    require(all(torch.equal(a,b) for a,b in zip(old_two,new_two)),'original entry terms changed')
    contributions=[];counts=[]
    for rank in range(8):
        pieces=[torch.tensor([(i+1)/100,(-1)**i*.2],dtype=torch.float64)*(100/48) for i in range(rank,48,8)]
        if rank==0:pieces.insert(0,torch.tensor([4.,-1.],dtype=torch.float64))
        if rank==1:pieces.insert(0,torch.tensor([-.5,3.],dtype=torch.float64))
        contributions.append(sum(pieces));counts.append(len(pieces))
    expected=sum(contributions)
    average=torch.stack([compensate_for_ddp(v) for v in contributions]).mean(0)
    clip=lambda v:v*min(1.,1/float(v.norm()))
    require(torch.allclose(average,expected) and not torch.allclose(average/8,expected) and
            not torch.allclose(torch.stack([8*v/n for v,n in zip(contributions,counts)]).mean(0),expected) and
            not torch.allclose(torch.stack([clip(8*v) for v in contributions]).mean(0),clip(expected)),
            'DDP compensation/rank-mean/preclip sensitivity')
    return dict(status='passed',support_gradient_ratio=10,gradient_max_abs_error=error,
                old_two_unchanged=True,compensation_and_clip_checks=True,
                support_coefficient_before=10,support_coefficient_after=100,
                analytic_expected_global_gradient=expected.tolist(),fixture_dtype='float64 analytic; runtime remains float32')


def prepare(output):
    require(not output.exists(),'occupied strong100 output')
    old=load_canonical_json(PRIOR/'inputs.json');proof=load_canonical_json(PRIOR/'receipt.json')
    require(proof['status']=='completed' and proof['numerical_admission']=='serial_dense48_global_gradient_v1' and
            proof['updates']==23 and proof['global_model_forwards']==1248,'accepted DDP path proof')
    for key in ('gradient_parity','two_step','launcher_exit'):
        ref=proof[key];require(file_hash(ref['path'])==ref['sha256'],'prior path evidence bytes')
    gradient=load_canonical_json(Path(proof['gradient_parity']['path']))
    smoke=load_canonical_json(Path(proof['two_step']['path']))
    require(gradient['status']=='passed' and gradient['raw_relative_l2']<=1e-5 and gradient['clipped_relative_l2']<=1e-5 and
            smoke['status']=='passed' and load_canonical_json(Path(proof['launcher_exit']['path']))['exit_code']==0,'prior DDP numerical path not admitted')
    fixture=coefficient_fixture()
    output.mkdir(parents=True,exist_ok=False);publish(output/'coefficient-fixture.json',fixture)
    keys=['cases','trajectories','model','train_source','qualifications','support_cases','support_image_ids','support_action_states',
          'added_dense_ids','support_preflight_sha256','rank_support_ids','bounds']
    packet={k:old[k] for k in keys}
    sources={path:sha for path,sha in old['source_files'].items() if '/dense48-parity-serial/' not in path and '/scores/' not in path}
    sources[str(PRIOR/'inputs.json')]=file_hash(PRIOR/'inputs.json');sources[str(PRIOR/'receipt.json')]=file_hash(PRIOR/'receipt.json')
    for key in ('gradient_parity','two_step','launcher_exit'):
        sources[proof[key]['path']]=proof[key]['sha256']
    for path,sha in sources.items():require(file_hash(path)==sha,f'frozen source changed: {path}')
    packet.update(schema_version='selective_preservation_strong.inputs.v1',source_files=sources,
        prior_path_proof=dict(path=str(PRIOR/'receipt.json'),sha256=file_hash(PRIOR/'receipt.json')),
        coefficient_fixture=dict(path=str(output/'coefficient-fixture.json'),sha256=file_hash(output/'coefficient-fixture.json')),
        numerical_admission=NUMERICAL_ADMISSION,optimizer=OPTIMIZER,lambda_kl=10.,lambda_support_kl=100.,updates=23,clip_gradient_norm=1.,
        objective='Originalmean2CE+10mean2KL unchanged; support100mean48KL; globally normalized image losses times8 before DDP average, clipafterreduction.')
    require(len(packet['support_image_ids'])==48 and packet['support_action_states']==5848 and packet['bounds']['reference_cache_bytes']==3678125640,'unchanged support48')
    for rank in range(8):rank_items(packet,rank)
    publish(output/'inputs.json',packet)
    files=[Path(__file__),Path(__file__).with_name('tests')/'test_selective_preservation_strong.py',CONFIG,
           Path(__file__).with_name('selective_preservation_dense.py'),Path(__file__).with_name('selective_preservation.py'),
           Path(__file__).with_name('train.py'),Path(__file__).with_name('runtime.py'),Path(__file__).with_name('entrance_ce.py')]
    files+=list(Path('src/qwen').glob('*.py'))+[Path('src/losses/token_scores.py'),Path('src/adapters/dora.py')]
    files += [Path('src/inference/inputs.py'), Path('src/inference/prompt.py'), Path('src/inference/image_plan.py')]
    records=[]
    for path in files:
        target=output/'effective_code'/str(path.resolve()).lstrip('/');target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
        records.append(dict(path=str(path.resolve()),staged=str(target),sha256=file_hash(target)))
    publish(output/'code_identity.json',dict(files=records,git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                                            git_status=subprocess.check_output(['git','status','--short'],text=True)))
    return packet


def execute_rank(output):
    import numpy as np
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from src.config.fingerprint import sha256_json
    from src.config.inference import load_research_infer_config
    from src.data import load_raw_examples
    from src.inference.runtime import assemble_frontend
    from src.qwen.native import prepare_replay
    from .runtime import bind_source256_language_dora, load_policy
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
        named, frozen = bind_source256_language_dora(
            model, expected_tensor_count=EXPECTED_TRAINABLE_TENSORS,
            expected_scalar_count=EXPECTED_TRAINABLE_SCALARS,
        )
        initial_adapter=_tensor_state_hash(named)
        initial_frozen=_tensor_state_hash(frozen)
        require(len(set(gather((initial_adapter,initial_frozen))))==1,'initial rank tensors differ')
        scorer=StrongScorer(model)
        ddp=DDP(scorer,device_ids=[local],output_device=local,broadcast_buffers=False,init_sync=False)
        # Freeze/version snapshot follows DDP construction, as in the repaired shared choreography.
        frozen_hash=_tensor_state_hash(frozen)
        require(frozen_hash==initial_frozen,'DDP construction changed frozen bytes')
        versions=[(p,p._version) for _,p in frozen]
        original=[p.detach().clone() for _,p in named]
        layout=_parameter_layout(named)
        publish(run/'trainable_layout.json',layout)
        gradient_layout=[];offset=0
        for entry in layout:
            gradient_layout.append({**entry,'offset':offset,'end':offset+entry['numel']});offset+=entry['numel']
        publish(run/'gradient_layout.json',gradient_layout)
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
        refstats=gather(dict(rank=rank,reference_count=len(refcards),cache_bytes=sum(c['cache_bytes'] for c in refcards),max_initial_abs_KL=max(abs(c['initial_KL']) for c in refcards)))
        require(sum(s['reference_count'] for s in refstats)==50 and sum(s['cache_bytes'] for s in refstats)==3678125640,'global reference bounds')
        def score_phase(step):
            barrier()
            result=None
            if rank==0:
              try:
                scores={}
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
                    score=summarize_logits(values,c['entrance']);score['target_id']=c['target_token_id'];scores[c['case_id']]=score
                publish(output/'scores'/f'step-{step:02d}.json',scores)
                result=dict(ok=True,scores=scores)
              except BaseException as exc:result=dict(ok=False,error=f'{type(exc).__name__}: {exc}')
            result=gather(result)[0]
            require(result['ok'],f"rank0 score phase failed: {result.get('error')}")
            return result
        initial_scores=score_phase(0)['scores']
        dose=[]
        step_gradient_records={}
        def save_gradient_snapshot(step,label):
            from safetensors.torch import save_file,load_file
            vector=torch.cat([p.grad.detach().flatten() for _,p in named]).cpu().contiguous()
            path=output/f'step-{step:02d}-{label}-gradient.safetensors'
            save_file({'gradient':vector},str(path))
            require(torch.equal(load_file(str(path))['gradient'],vector),'gradient disk reload')
            return dict(path=str(path),sha256=file_hash(path),l2_float64=float(vector.double().norm()))
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
                counters['support_KL_trajectories']+=int(item['kind']=='support')
                local_losses.append(dict(key=item['key'],kind=item['kind'],global_weighted_loss=float(loss.detach())/8,
                                         CE=float(ce) if ce is not None else None,mean_KL=float(kl),states=len(item['positions'])))
                del loss,ce,kl
            counters['ddp_synchronized_backwards']+=1
            finite=finite and all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for _,p in named)
            require(all_true(finite),'nonfinite/missing globally reduced gradients')
            require(all(p.grad is None and not p.requires_grad for _,p in frozen),'frozen gradients')
            gradient_hash=_tensor_state_hash([(n,p.grad) for n,p in named])
            gradient_hashes=gather(gradient_hash)
            if step<=2:
                snapshot=None
                if rank==0:
                  try:snapshot=dict(ok=True,record=save_gradient_snapshot(step,'raw'))
                  except BaseException as exc:snapshot=dict(ok=False,error=f'{type(exc).__name__}: {exc}')
                snapshot=gather(snapshot)[0]
                require(snapshot['ok'],f"raw gradient snapshot failed: {snapshot.get('error')}")
                step_gradient_records['raw']=snapshot['record']
            else:require(len(set(gradient_hashes))==1,'reduced gradient differs by rank')
            raw_norm=float(torch.nn.utils.clip_grad_norm_([p for _,p in named],1.,error_if_nonfinite=True,foreach=False))
            require(math.isfinite(raw_norm) and raw_norm>0,'invalid global gradient norm')
            clipped_norm=math.sqrt(sum(float(p.grad.double().square().sum()) for _,p in named))
            require(clipped_norm<=1.000001,'global clip bound')
            if step<=2:
                snapshot=None
                if rank==0:
                  try:snapshot=dict(ok=True,record=save_gradient_snapshot(step,'clipped'))
                  except BaseException as exc:snapshot=dict(ok=False,error=f'{type(exc).__name__}: {exc}')
                snapshot=gather(snapshot)[0]
                require(snapshot['ok'],f"clipped gradient snapshot failed: {snapshot.get('error')}")
                step_gradient_records['clipped']=snapshot['record']
            optimizer.step();counters['updates']=step
            require(all(p._version==v for p,v in versions),'frozen version changed')
            require(all(not x.requires_grad and x.grad_fn is None and x._version==refversions[k] for k,x in refs.items()),'reference cache changed')
            adapter_hash=_tensor_state_hash(named);opt_hash=optimizer_hash(optimizer,named)
            state_hashes=gather((adapter_hash,opt_hash))
            if step<=2:
                snapshot=None
                if rank==0:
                  try:
                    optimizer_path=output/f'step-{step:02d}-optimizer.pt';torch.save(optimizer.state_dict(),optimizer_path)
                    restored=torch.load(optimizer_path,map_location='cpu',weights_only=True);live=optimizer.state_dict()
                    require(restored['param_groups']==live['param_groups'],'optimizer config reload')
                    require(all(torch.equal(restored['state'][k][field],value.cpu()) for k,st in live['state'].items() for field,value in st.items()),'optimizer tensor reload')
                    diagnostic_adapter=_save_adapter_only(model,source_root=Path(config.adapter.path),output=output/f'step-{step:02d}-diagnostic-adapter')
                    state_path=output/f'step-{step:02d}-state-snapshot.json'
                    publish(state_path,dict(mechanical_diagnostic_only=True,raw_gradient=step_gradient_records['raw'],clipped_gradient=step_gradient_records['clipped'],
                        optimizer=dict(path=str(optimizer_path),sha256=file_hash(optimizer_path)),adapter=diagnostic_adapter,
                        rank_raw_gradient_hashes=gradient_hashes,rank_adapter_optimizer_hashes=[list(x) for x in state_hashes],
                        raw_gradient_norm=raw_norm,clipped_gradient_norm=clipped_norm,gradient_layout_sha256=file_hash(run/'gradient_layout.json')))
                    passed=len(set(gradient_hashes))==1 and len(set(state_hashes))==1
                    agreement=dict(status='passed' if passed else 'failed',update=step,numerical_admission=NUMERICAL_ADMISSION,
                        exact_raw_gradient_rank_identity=len(set(gradient_hashes))==1,exact_adapter_optimizer_rank_identity=len(set(state_hashes))==1,
                        state_snapshot=dict(path=str(state_path),sha256=file_hash(state_path)),prior_path_proof=packet['prior_path_proof'],coefficient_fixture=packet['coefficient_fixture'])
                    publish(output/f'step-{step:02d}-state-agreement.json',agreement)
                    snapshot=dict(ok=True,passed=passed)
                  except BaseException as exc:snapshot=dict(ok=False,error=f'{type(exc).__name__}: {exc}')
                snapshot=gather(snapshot)[0]
                require(snapshot['ok'],f"state snapshot failed: {snapshot.get('error')}")
                require(snapshot['passed'],'rank state disagreement; snapshots preserved')
            else:require(len(set(state_hashes))==1,'rank adapter/optimizer differs')
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
            step2_KL_passed=any(s['max_KL']>1e-8 for s in perrank)
            row=dict(update=step,rank=rank,local_losses=local_losses,gradient_hash=gradient_hash,adapter_hash=adapter_hash,optimizer_hash=opt_hash,
                     raw_gradient_norm=raw_norm,clipped_gradient_norm=clipped_norm,source_parameter_delta_l2=total_movement,
                     counters=dict(counters),seconds=time.monotonic()-started)
            publish(run/f'update-{step:02d}.json',row)
            if rank==0:
                dose.append(dict(update=step,ranks=perrank,final_scores=final_scores))
                publish(output/f'update-{step:02d}.json',dose[-1])
                if step==2:
                    publish(output/'two-step-smoke.json',dict(status='passed' if step2_KL_passed else 'failed',updates=[1,2],initial_reference_stats=refstats,
                        initial_KL_zero=all(x['max_initial_abs_KL']==0 for x in refstats),initial_reference_count=sum(x['reference_count'] for x in refstats),
                        step2_max_KL_by_rank={str(x['rank']):x['max_KL'] for x in perrank},numerical_admission=NUMERICAL_ADMISSION,
                        prior_path_proof=packet['prior_path_proof'],coefficient_fixture=packet['coefficient_fixture'],
                        step1_state_agreement_sha256=file_hash(output/'step-01-state-agreement.json'),step2_state_agreement_sha256=file_hash(output/'step-02-state-agreement.json'),
                        reduced_gradient_adapter_optimizer_rank_identity=True,frozen_bytes_unchanged=True,source_parameter_delta_l2=total_movement))
                    if step2_KL_passed:print('TWO_STEP_SMOKE_COMPLETED',flush=True)
            # All ranks confirm rank0 has published the gate before beginning update3.
            if step==2:
                barrier();require(step2_KL_passed,'step2 nonzero KL not exercised; evidence saved')
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
            provisional=dict(schema_version='selective_preservation_strong.training.v1',status='unsealed',adapter=adapter,
                source_adapter=packet['model']['current_adapter'],source_embedding=packet['model']['source_embedding'],
                config=config.model_dump(mode='json'),cases=packet['cases'],final_scores=final_scores,initial_scores=initial_scores,
                updates=23,stop_reason='fixed_steps',optimizer=OPTIMIZER,lambda_kl=10.,lambda_support_kl=100.,clip_gradient_norm=1.,
                support_image_ids=packet['support_image_ids'],support_action_states=5848,support_preflight_sha256=packet['support_preflight_sha256'],
                objective=packet['objective'],trainable_layout=layout,dose=dose,rank_training_states=ranks,
                inputs_sha256=file_hash(output/'inputs.json'),code_identity_sha256=file_hash(output/'code_identity.json'),
                numerical_admission=NUMERICAL_ADMISSION,prior_path_proof=packet['prior_path_proof'],coefficient_fixture=packet['coefficient_fixture'],
                two_step=dict(path=str(output/'two-step-smoke.json'),sha256=file_hash(output/'two-step-smoke.json')),
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
    exit_path=output/'launcher_exit.json';require(load_canonical_json(exit_path)['exit_code']==0,'launcher failed')
    require({p.name for p in (output/'ranks').iterdir() if p.is_dir()}=={f'rank{r}' for r in range(8)},'rank coverage')
    terminals=[];refs=[]
    for rank in range(8):
        path=output/'ranks'/f'rank{rank}'/'terminal.json';t=load_canonical_json(path)
        require(t['rank']==rank and t['status']=='completed' and t['updates']==23 and t['model_forwards']==CEILINGS[rank] and
                t['counters']['support_KL_trajectories']==138,'rank terminal completeness')
        terminals.append(t);refs.append(dict(rank=rank,path=str(path),sha256=file_hash(path)))
    require(len({t['state']['adapter_hash'] for t in terminals})==len({t['state']['optimizer_hash'] for t in terminals})==1,'final rank states')
    p=load_canonical_json(output/'provisional.json')
    require(p['numerical_admission']==NUMERICAL_ADMISSION and p['lambda_support_kl']==100 and p['lambda_kl']==10,'strong100 numerical recipe')
    for ref in (p['prior_path_proof'],p['coefficient_fixture'],p['two_step']):require(file_hash(ref['path'])==ref['sha256'],'admission binding')
    fixture=load_canonical_json(Path(p['coefficient_fixture']['path']));smoke=load_canonical_json(Path(p['two_step']['path']))
    require(fixture['status']=='passed' and fixture['support_gradient_ratio']==10 and fixture['old_two_unchanged'] and
            fixture['compensation_and_clip_checks'],'coefficient fixture')
    require(smoke['status']=='passed' and smoke['updates']==[1,2] and smoke['initial_KL_zero'] and smoke['initial_reference_count']==50 and
            any(k>1e-8 for k in smoke['step2_max_KL_by_rank'].values()) and smoke['reduced_gradient_adapter_optimizer_rank_identity'] and
            smoke['frozen_bytes_unchanged'],'current two-step gate')
    for ident in (p['adapter'],p['source_embedding']):
        for f in ident['files']:require(file_hash(Path(ident['root'])/f['relative_path'])==f['sha256'],'checkpoint bytes')
    actual_support=sum(t['counters']['support_KL_trajectories'] for t in terminals)
    require(actual_support==1104 and sum(t['model_forwards'] for t in terminals)==1248,'global execution counts')
    receipt={**p,'status':'completed','rank_terminals':refs,'launcher_exit':dict(path=str(exit_path),sha256=file_hash(exit_path)),
        'actual_support_uses':actual_support,'global_model_forwards':1248,'provisional_sha256':file_hash(output/'provisional.json'),
        'resources':dict(model_loads=8,reference_forwards=sum(t['counters']['reference_forwards'] for t in terminals),
            train_forwards=sum(t['counters']['train_forwards'] for t in terminals),entry_score_forwards=sum(t['counters']['entry_score_forwards'] for t in terminals),
            cumulative_rank_model_seconds=sum(t['cumulative_model_seconds'] for t in terminals),max_rank_model_seconds=max(t['cumulative_model_seconds'] for t in terminals),
            reference_cache_bytes=sum(t['state']['reference_cache_bytes'] for t in terminals),
            peak_cuda_allocated_bytes_max_rank=max(t['state']['peak_cuda_allocated_bytes'] for t in terminals),
            peak_cuda_reserved_bytes_max_rank=max(t['state']['peak_cuda_reserved_bytes'] for t in terminals),
            peak_rss_bytes_max_rank=max(t['state']['peak_rss_bytes'] for t in terminals))}
    publish(output/'receipt.json',receipt);return receipt


def launch(output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='0,1,2,3,4,5,6,7' and not (output/'launcher_owner.json').exists(),'single eight-GPU launcher')
    command=[sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc-per-node=8',
             '-m','probes.dora_owner_learning.selective_preservation_strong','rank','--output',str(output)]
    publish(output/'launcher_owner.json',dict(pid=os.getpid(),command=command,started=time.time(),
        processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],text=True)))
    started=time.monotonic()
    with (output/'torchrun.log').open('x') as stream:result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=False)
    publish(output/'launcher_exit.json',dict(exit_code=result.returncode,elapsed_seconds=time.monotonic()-started,finished=time.time()))
    if result.returncode:return result.returncode
    receipt=finalize(output);print(json.dumps(dict(status='completed',resources=receipt['resources'])));return 0


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','rank','launch','verify']);parser.add_argument('--output',type=Path,default=OUTPUT);args=parser.parse_args()
    if args.command=='prepare':
        p=prepare(args.output);print(json.dumps(dict(support=len(p['support_image_ids']),lambda_support=p['lambda_support_kl'],coefficient_fixture=p['coefficient_fixture'])))
    elif args.command=='rank':execute_rank(args.output)
    elif args.command=='launch':raise SystemExit(launch(args.output))
    else:
        r=load_canonical_json(args.output/'receipt.json');require(r['status']=='completed' and r['actual_support_uses']==1104 and r['numerical_admission']==NUMERICAL_ADMISSION,'receipt completeness')
        for ref in r['rank_terminals']+[r['launcher_exit'],r['prior_path_proof'],r['coefficient_fixture'],r['two_step']]:require(file_hash(ref['path'])==ref['sha256'],'sealed evidence binding')
        for ident in (r['adapter'],r['source_embedding']):
            for f in ident['files']:require(file_hash(Path(ident['root'])/f['relative_path'])==f['sha256'],'sealed adapter bytes')
        print(json.dumps(dict(status=r['status'],updates=r['updates'],resources=r['resources'],final_margins={c:s['A_vs_best_other_margin'] for c,s in r['final_scores'].items()})))


if __name__=='__main__':main()
