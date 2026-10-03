"""Fixed-bank A/B preparation, released runtime and existing evaluator consumers."""
from __future__ import annotations
import argparse
from collections import defaultdict
from contextlib import nullcontext
import json
import math
from pathlib import Path
import torch
from probes import online_row_credit as o
from probes.full_label_fit import experiment as fit
from src.qwen.inspection import resolve_text_stack
from .bank import UNIT, OUT, require, read, write, checked, verify_bank, bank_costs, execution_jobs, selection_cases
from .objective import SEED, TrainableRows, make_head, auxiliary, clip_separately, save_head

def source_paths():
    return [*o.source_paths(True),*sorted(str(p) for p in Path('probes/pre_row_detection_aux').glob('*.py')),
        'probes/online_row_credit_owner.py']


def packet_commands(root,bank,bank_sha256,selection,selection_sha256,anchor,updates):
    """Exact serial stages; the existing owner substitutes the release digest."""
    from probes.online_row_credit_owner import PRE_ROW_STAGES
    commands=[]
    for arm,stage in PRE_ROW_STAGES:
        prefix=['python','-m','torch.distributed.run','--standalone','--nproc-per-node=8','--module','probes.pre_row_detection_aux'] if stage in ('run','evaluate') else ['python','-m','probes.pre_row_detection_aux']
        output=root/('offline-results.json' if stage=='offline' else arm if stage in ('run','arm-readback') else 'evaluation-'+arm)
        command=[*prefix,stage,'--bank',str(bank),'--bank-sha256',bank_sha256,'--output',str(output)]
        if stage in ('run','arm-readback'):command += ['--arm',arm,'--updates',str(updates)]
        if stage in ('evaluate','eval-readback','offline'):command += ['--selection',str(selection),'--selection-sha256',selection_sha256]
        if stage=='evaluate':command += ['--checkpoint',str(anchor if arm=='zero' else root/arm/f'checkpoint-{updates}')]
        if stage in ('run','evaluate'):command += ['--release',str(root/'lead-release.json'),'--release-sha256','LEAD_RELEASE_SHA256']
        if stage=='offline':command += ['--zero',str(root/'evaluation-zero'),'--arm-a',str(root/'evaluation-A'),'--arm-b',str(root/'evaluation-B')]
        commands.append(dict(arm=arm,stage=stage,argv=command))
    return commands


def prepare_packet(args):
    import subprocess
    import sys
    from importlib.metadata import version
    from .bank import positive_rows,opener_position
    from .objective import INITIALIZER
    require(args.updates in (1,16),'only proposed one-update qualification or frozen sixteen-update packet')
    bank=verify_bank(args.bank,args.bank_sha256)
    cases=selection_cases(args.selection,args.selection_sha256,bank,args.bank_sha256)
    tokenizer=o.r.frontend().tokenizer
    images={i['image_id']:i for i in bank['images']};records={r['image_id']:r for r in bank['records']};plans={p['image_id']:p for p in bank['plans']}
    for i in images:
        require(o.completion_credit(images[i],records[i],tokenizer,records[i]['producer'],'treatment',True,o.RESTORED_M_WEIGHTING)==plans[i],'current plan projection drift')
    projected=[]
    for k,job in enumerate(bank['jobs']):
        i=job['image_id']
        for row_index,(row,sequence,weight,kind) in enumerate(positive_rows(images[i],records[i],plans[i],tokenizer,job)):
            position=opener_position(sequence,row,len(records[i]['prompt_token_ids']),tokenizer)
            saved=bank['rows'][len(projected)]
            require(saved['job_index']==k and saved['row_index']==row_index and saved['position']==position and saved['prefix']==list(sequence.input_ids[:position+1]) and saved['weight']==weight and saved['kind']==kind,'saved opener/weight projection drift')
            projected.append(saved)
    require(len(projected)==580,'positive row drift')
    jobs=execution_jobs(bank,tokenizer)
    hidden_size=read(Path(o.p.load(o.p.POLICY)['base_model'])/'config.json')['text_config']['hidden_size']
    costs=bank_costs(bank,jobs,cases,args.updates,hidden_size)
    initial_head=make_head(hidden_size,len(bank['classes']))
    initial_head_tensors={name:dict(shape=list(value.shape),dtype=str(value.dtype),sha256=o.p.tensor_hash(value)) for name,value in initial_head.state_dict().items()}
    root=args.output.resolve();require(root.is_relative_to(OUT.resolve()),'packet root outside unit output')
    root.mkdir(parents=True,exist_ok=False)
    commands=packet_commands(root,args.bank.resolve(),args.bank_sha256,args.selection.resolve(),args.selection_sha256,Path(bank['checkpoint']),args.updates)
    runtime={name:version(name) for name in ('torch','transformers','vllm','peft','safetensors')}
    test_sources={p:fit.sha(Path(p)) for p in ('tests/probes/test_pre_row_detection_aux.py','tests/probes/test_online_row_credit_owner.py')}
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    qual=dict(schema='pre-row-fixed-bank-qualification-v1',status='CPU_candidate_needs_clean_source_native_qualification',
        bank=str(args.bank.resolve()),bank_sha256=args.bank_sha256,selection=str(args.selection.resolve()),selection_sha256=args.selection_sha256,
        updates=args.updates,source=dict(commit=commit,files=[]),source_paths=source_paths(),
        sha256={**bank['input_sha256'],str(args.bank.resolve()):args.bank_sha256,str(args.selection.resolve()):args.selection_sha256},
        runtime=runtime,decoder_runtime_identity=fit.decoder_runtime_identity(),test_source_sha256=test_sources,
        costs=costs,execution_jobs=jobs,initializer=INITIALIZER,initial_head_tensors=initial_head_tensors,directly_supervised_owners=568,
        inherited_unsupervised_owners=[[4134,-99],[7511,-167]],primary_GT_denominator=570,native_qualified=False)
    write(root/'qualification.json',qual);write(root/'argv.json',commands)
    native=[]
    for row in commands:
        if row['stage'] in ('run','evaluate'):
            tail=row['argv'][row['argv'].index('probes.pre_row_detection_aux')+1:]
            native.append([sys.executable,'-m','probes.pre_row_detection_aux',*tail])
    release=dict(mode='pre-row-aux',unit_id=UNIT.name,native_released=False,updates=args.updates,
        total_wall_ceiling_seconds=costs['estimates']['whole_owner_wall_seconds'],owner_root=str(root),
        bank_sha256=args.bank_sha256,selection_sha256=args.selection_sha256,source_commit=commit,
        lead_thread='01a0fdd8-26b6-7240-ab56-f021c05f3445',worker_thread='01a100b7-796d-7530-b29d-94abe1af2170',
        argv_sha256=fit.sha(root/'argv.json'),qualification_sha256=fit.sha(root/'qualification.json'),runtime=runtime,
        bindings={'probes/online_row_credit_owner.py':fit.sha(Path('probes/online_row_credit_owner.py'))},exact_invocations=native,
        stop='CPU candidate only; lead must qualify clean frozen execution source and explicitly release exact packet')
    write(root/'release-candidate.json',release)
    return dict(status='CPU_candidate',root=str(root),costs=costs,bank_sha256=args.bank_sha256,selection_sha256=args.selection_sha256,native_released=False)


def replay_training_loss(q, model, batch, image, record, plan, vocab, job, rows, classes, denominator, head=None):
    selected = [r for r in rows if r['eligible']]
    capture = TrainableRows(resolve_text_stack(q.model).norm, [r['position'] for r in selected]) if head is not None and selected else None
    with capture if capture is not None else nullcontext():
        base, evidence = o.forward(q, model, batch, image, record, plan, None, vocab, job['branch'],
            geometry_weight=.1, branch_index=job.get('branch_index'), correction_arm='treatment', duplicate_weight=1,
            replay_device_type=next(q.model.parameters()).device.type)
    evidence['base_loss'] = float(base.detach())
    if capture is None:
        evidence['auxiliary'] = dict(loss=0., rows=0, coefficient=.1 if head is not None else 0.)
        return base, evidence
    aux, detail = auxiliary(head, capture.hidden, selected, classes, denominator)
    evidence['auxiliary'] = dict(detail, loss=float(aux.detach()), coefficient=.1,
        positions=[r['position'] for r in selected], row_ids=[r['row_id'] for r in selected])
    loss = base + .1*aux
    evidence['loss'] = float(loss.detach())
    return loss, evidence


def runtime_release(args):
    """Fail closed before model/device access; a candidate is never a release."""
    from importlib.metadata import version
    import subprocess
    import sys
    release = checked(args.release, args.release_sha256)
    require(release.get('native_released') is True and release['unit_id'] == UNIT.name, 'native not released')
    require(release['bank_sha256'] == args.bank_sha256, 'release bank drift')
    if args.command=='evaluate':require(release['selection_sha256']==args.selection_sha256,'release selection drift')
    invocation = [sys.executable,'-m','probes.pre_row_detection_aux',*sys.argv[1:]]
    invocation[invocation.index('--release-sha256')+1] = 'LEAD_RELEASE_SHA256'
    require(invocation in release['exact_invocations'], 'invocation not released')
    require(subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip() == release['source_commit'], 'release source drift')
    for package, expected in release['runtime'].items():
        require(version(package) == expected, f'runtime drift: {package}')
    # The owner admits a fresh stage before launching ranks; each rank uses exclusive mkdir.
    require(args.output.resolve().is_relative_to(OUT.resolve()), 'output ownership')
    return release


def run_arm(args):
    import os
    import time
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from src.losses.vocab import build_token_vocabulary_groups
    from src.artifacts.git_identity import capture_source_identity, verify_source_identity
    release = runtime_release(args)
    bank = verify_bank(args.bank,args.bank_sha256)
    require(args.updates in (1,16) and release['updates']==args.updates, 'dose not released')
    require(int(os.environ['WORLD_SIZE'])==8, 'expected eight ranks')
    rank = int(os.environ['RANK']); local_rank = int(os.environ['LOCAL_RANK'])
    sources = source_paths()
    source = capture_source_identity(sources)
    o.verify_anchor_payload(Path(bank['checkpoint']),bank['checkpoint_manifest_sha256'])
    begin = time.monotonic(); out = args.output / f'rank-{rank}'
    out.mkdir(parents=True,exist_ok=False)
    write(out/'entry.json',dict(source=source,bank_sha256=args.bank_sha256,arm=args.arm,pid=os.getpid()))
    torch.cuda.set_device(local_rank);torch.manual_seed(SEED)
    dist.init_process_group('nccl')
    try:
        q,delta,composition = o.p.compose(Path(bank['checkpoint']))
        require(q.model.config.text_config.attention_dropout==0, 'attention dropout drift')
        require(all(getattr(m,'p',0)==0 for n,m in q.model.named_modules() if 'lora_dropout' in n),'adapter dropout drift')
        write(out/'composition.json',composition)
        o.set_checkpointing(q.model,False)
        q.model.train()
        params = [p for p in q.model.parameters() if p.requires_grad]
        require(len(params)==590,'backbone trainables drift')
        delta_ids = {id(p) for p in delta.delta_tensors().values()}
        optimizer = torch.optim.AdamW([dict(params=[p for p in params if id(p) not in delta_ids],lr=1e-5),
            dict(params=list(delta.delta_tensors().values()),lr=5e-6)],betas=(.9,.999),eps=1e-8,weight_decay=0)
        model = DistributedDataParallel(q.model,device_ids=[local_rank],broadcast_buffers=False)
        head = make_head(q.model.config.text_config.hidden_size,len(bank['classes'])).to('cuda') if args.arm=='B' else None
        head_optimizer = torch.optim.AdamW(head.parameters(),lr=1e-3,betas=(.9,.999),eps=1e-8,weight_decay=0) if head is not None else None
        vocab = build_token_vocabulary_groups(q.token_identity,tokenizer=q.tokenizer)
        images = {x['image_id']:x for x in bank['images']};records = {x['image_id']:x for x in bank['records']};plans = {x['image_id']:x for x in bank['plans']}
        local_jobs = [dict(j,job_index=k) for k,j in enumerate(bank['jobs']) if j['rank']==rank]
        batches = {i:o.native_batch(q,records[i]) for i in sorted(images)[rank::8]}
        rows_by_job = defaultdict(list)
        for row in bank['rows']:rows_by_job[row['job_index']].append(row)
        for update in range(1,args.updates+1):
            optimizer.zero_grad(set_to_none=True)
            if head_optimizer is not None:head_optimizer.zero_grad(set_to_none=True)
            forwards = o.r.accumulate_family_step(model,local_jobs,lambda j: replay_training_loss(q,model,batches[j['image_id']],
                images[j['image_id']],records[j['image_id']],plans[j['image_id']],vocab,j,rows_by_job[j['job_index']],
                bank['classes'],bank['image_denominators'][str(j['image_id'])],head))
            if head is not None:
                # Head stays outside the deployed model; match DDP's mean reduction explicitly.
                for parameter in head.parameters():
                    if parameter.grad is None:parameter.grad=torch.zeros_like(parameter)
                    dist.all_reduce(parameter.grad);parameter.grad.div_(8)
            norms = {n:float(p.grad.float().norm()) if p.grad is not None else None for n,p in q.model.named_parameters() if p.requires_grad}
            head_norms = {n:float(p.grad.float().norm()) for n,p in head.named_parameters()} if head is not None else {}
            clips = clip_separately(params,list(head.parameters()) if head is not None else [])
            optimizer.step()
            if head_optimizer is not None:head_optimizer.step()
            require(all(torch.isfinite(p).all() for p in params),'nonfinite backbone')
            if head is not None:require(all(torch.isfinite(p).all() for p in head.parameters()),'nonfinite head')
            write(out/f'update-{update}.json',dict(update=update,arm=args.arm,forwards=forwards,
                gradient_norms=norms,head_gradient_norms=head_norms,preclip=clips,lrs=[g['lr'] for g in optimizer.param_groups],
                head_lr=1e-3 if head is not None else None,seconds=time.monotonic()-begin,bank_sha256=args.bank_sha256))
        if rank==0:
            o.p.save_checkpoint(q,delta,args.output/f'checkpoint-{args.updates}')
            if head is not None:save_head(head,args.output/'training-head.pt',args.bank_sha256,bank['classes'])
        dist.barrier()
        verify_source_identity(source,required_paths=sources)
        write(out/'complete.json',dict(status='complete',source=source,bank_sha256=args.bank_sha256,arm=args.arm,updates=args.updates,
            wall_seconds=time.monotonic()-begin,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),
            artifacts={p.name:fit.sha(p) for p in out.glob('*.json')}))
    finally:
        dist.destroy_process_group()


def evaluate(args):
    """Ordinary head-free vLLM deployment; conditional prefixes are a separate pool."""
    import os
    import time
    from src.qwen.vllm_rollout import VllmDoraRollout
    from src.artifacts.git_identity import capture_source_identity, verify_source_identity
    release = runtime_release(args)
    bank = verify_bank(args.bank,args.bank_sha256)
    cases=selection_cases(args.selection,args.selection_sha256,bank,args.bank_sha256)
    require(release['selection_sha256']==args.selection_sha256,'conditional subset not frozen')
    require(int(os.environ['WORLD_SIZE'])==8, 'expected eight ranks')
    rank=int(os.environ['RANK']);local_rank=int(os.environ['LOCAL_RANK'])
    sources=source_paths();source=capture_source_identity(sources)
    if args.checkpoint.resolve()==Path(bank['checkpoint']).resolve():
        o.verify_anchor_payload(args.checkpoint,bank['checkpoint_manifest_sha256'])
    out=args.output/f'rank-{rank}';out.mkdir(parents=True,exist_ok=False);begin=time.monotonic()
    q=o.r.frontend();records={r['image_id']:r for r in bank['records']}
    ids=sorted(records)[rank::8];requests=o.vllm_requests(q,records,ids)
    selected=[row for k,row in enumerate(cases) if k%8==rank]
    snapshot=fit.sha(args.checkpoint / ('identity.json' if (args.checkpoint/'identity.json').exists() else 'inference_payload_manifest.json'))
    rollout=VllmDoraRollout(base_model=q.base_model_path,checkpoint=args.checkpoint,identity=snapshot,log_path=out/'vllm.log',device=local_rank,trainer_rank=rank)
    try:
        generation_start=time.monotonic()
        results=rollout.generate(requests,budgets=[3084]*len(ids),eos_token_id=q.tokenizer.convert_tokens_to_ids('<|im_end|>'),pad_token_id=q.tokenizer.pad_token_id,identity=snapshot)
        generation_seconds=time.monotonic()-generation_start
        for i,result in zip(ids,results,strict=True):
            raw=records[i]
            record=dict({k:v for k,v in raw.items() if not k.startswith('generation_')},token_ids=list(result.token_ids),text=q.tokenizer.decode(result.token_ids,skip_special_tokens=False),
                generated_tokens=len(result.token_ids),stop_reason=result.stop_reason,raw_logprobs=None)
            # The evaluator consumes only decoded tokens; preserve new deployment identity distinctly.
            record.pop('raw_identity');record.pop('producer')
            record.update(evaluation_checkpoint=str(args.checkpoint),snapshot=snapshot,bank_sha256=args.bank_sha256,
                generation_batch_seconds=generation_seconds,generation_batch_size=len(ids),generation_seconds=generation_seconds/len(ids))
            write(out/f'{i}.json',record)
        conditional=[]
        if selected:
            exact_requests=o.vllm_requests(q,records,[r['image_id'] for r in selected])
            # vLLM expands the chat's single media placeholder; HF bank IDs are already expanded.
            results=rollout.generate_exact(exact_requests,chat_token_ids=[q.tokenizer.encode(r.chat_text,add_special_tokens=False) for r in exact_requests],
                extensions=[r['prefix'][len(records[r['image_id']]['prompt_token_ids']):] for r in selected],budgets=[64]*len(selected),
                eos_token_id=q.tokenizer.convert_tokens_to_ids('<|im_end|>'),pad_token_id=q.tokenizer.pad_token_id,identity=snapshot,vocab_size=len(q.tokenizer))
            for row,result in zip(selected,results,strict=True):
                conditional.append(dict(row_id=row['row_id'],image_id=row['image_id'],owner=row['owner'],history_kind=row['history_kind'],
                    prefix_sha256=row['prefix_sha256'],token_ids=list(result['token_ids']),text=q.tokenizer.decode(result['token_ids'],skip_special_tokens=False),
                    stop_reason=result['stop_reason'],generated_tokens=len(result['token_ids'])))
        write(out/'conditional.json',conditional)
    finally:
        rollout.close()
        write(out/'vllm-operations.json',rollout.receipts)
    verify_source_identity(source,required_paths=sources)
    write(out/'complete.json',dict(status='complete',source=source,bank_sha256=args.bank_sha256,snapshot=snapshot,
        wall_seconds=time.monotonic()-begin,selection_sha256=args.selection_sha256,artifacts={f'{i}.json':fit.sha(out/f'{i}.json') for i in ids},
        conditional_sha256=fit.sha(out/'conditional.json'),operations_sha256=fit.sha(out/'vllm-operations.json')))


def evaluation_readback(path, bank, bank_sha256, cases,selection_sha256):
    records=o.frozen_records(path,[i['image_id'] for i in bank['images']])
    conditional=[]
    for rank in range(8):
        out=path/f'rank-{rank}';receipt=read(out/'complete.json')
        require(receipt['bank_sha256']==bank_sha256,'evaluation bank identity drift')
        require(receipt['selection_sha256']==selection_sha256,'evaluation selection identity drift')
        conditional.extend(checked(out/'conditional.json',receipt['conditional_sha256']))
        checked(out/'vllm-operations.json',receipt['operations_sha256'])
    require(sorted(r['row_id'] for r in conditional)==sorted(r['row_id'] for r in cases),'conditional coverage drift')
    # Only the root finalizer receipt is derived; all other JSON remains frozen payload.
    manifest=path/'frozen.json';readback=path/'readback.json'
    frozen={str(p.relative_to(path)):fit.sha(p) for p in sorted(path.rglob('*.json')) if p not in (manifest,readback)}
    if manifest.exists():require(read(manifest)==frozen,'evaluation frozen payload drift')
    else:
        require(not readback.exists(),'evaluation frozen manifest missing')
        write(manifest,frozen)
    if readback.exists():
        require(read(readback)==dict(status='complete',bank_sha256=bank_sha256,ordinary_requests=len(records),
            conditional_requests=len(conditional),frozen_sha256=fit.sha(manifest)),'evaluation readback receipt drift')
    return records,conditional


def arm_readback(output, bank, bank_sha256, arm, updates):
    forwards=[]
    jobs=execution_jobs(bank,o.r.frontend().tokenizer)
    for rank in range(8):
        out=output/f'rank-{rank}';receipt=read(out/'complete.json')
        require(receipt['status']=='complete' and receipt['bank_sha256']==bank_sha256 and receipt['arm']==arm and receipt['updates']==updates,'arm completion drift')
        for name,digest in receipt['artifacts'].items():checked(out/name,digest)
        for update in range(1,updates+1):
            value=read(out/f'update-{update}.json')
            expected=[j for j in jobs if j['rank']==rank]
            require(len(value['forwards'])==len(expected),'schedule count drift')
            for observed,job in zip(value['forwards'],expected,strict=True):
                require(observed['image_id']==job['image_id'] and observed['branch']==job['branch'] and observed['image_weight']==job['weight'] and observed['sync']==job['sync'],'schedule identity drift')
                require(observed['tokens']==job['tokens'] and observed['positions']==job['positions'] and observed['input_sha256']==job['input_sha256'],'replay input identity drift')
                require(observed['correction']['plan_sha256']==o.identity(next(p for p in bank['plans'] if p['image_id']==job['image_id'])),'persisted credit identity drift')
                require(math.isfinite(observed['loss']) and math.isfinite(observed['base_loss']),'nonfinite persisted loss')
            forwards.extend(value['forwards'])
    checkpoint=output/f'checkpoint-{updates}'
    for relative,digest in read(checkpoint/'identity.json').items():
        require(fit.sha(checkpoint/relative)==digest,'deployed checkpoint identity drift')
    require(not any('head' in p.name for p in checkpoint.rglob('*')),'training head in deployed export')
    if arm=='B':
        payload=torch.load(output/'training-head.pt',map_location='cpu',weights_only=True)
        require(payload['bank_sha256']==bank_sha256 and payload['classes']==bank['classes'],'persisted head identity drift')
    else:require(not (output/'training-head.pt').exists(),'A has auxiliary head')
    result=dict(status='complete',evidence='persisted_artifact_verification',arm=arm,updates=updates,forwards=len(forwards),bank_sha256=bank_sha256,
        checkpoint_identity_sha256=fit.sha(checkpoint/'identity.json'))
    write(output/'readback.json',result)
    return result


def conditional_metrics(bank, records):
    """Score only the first free completed row; fixed history never earns credit."""
    by_row={r['row_id']:r for r in bank['rows']};by_image={i['image_id']:i for i in bank['images']};out=[]
    for result in records:
        row=by_row[result['row_id']]; image=by_image[row['image_id']]
        require(result['prefix_sha256']==row['prefix_sha256'] and result['history_kind']==row['history_kind'],'conditional identity drift')
        probe=dict(image,request_id=f"conditional-{row['row_id']}",text='<|object_ref_start|>'+result['text'])
        parsed=o.r.parse(probe)
        first=parsed.predictions[0] if parsed.predictions and parsed.predictions[0]['generated_order']==0 and parsed.predictions[0]['char_start']==0 else None
        correct=bool(first and first['description']==row['class'] and o.iou_xyxy(first['coord_bins'],row['box'])>=.5)
        out.append(dict(row_id=row['row_id'],image_id=row['image_id'],owner=row['owner'],history_kind=row['history_kind'],
            first_row_correct=correct,first_row=first,stop_reason=result['stop_reason'],generated_tokens=result['generated_tokens']))
    return dict(rows=out,by_history={h:dict(rows=sum(r['history_kind']==h for r in out),correct=sum(r['first_row_correct'] for r in out if r['history_kind']==h)) for h in ('actual','synthetic')})


def offline(args):
    bank=verify_bank(args.bank,args.bank_sha256)
    cases=selection_cases(args.selection,args.selection_sha256,bank,args.bank_sha256)
    frozen={};conditional={}
    for name,path in (('zero',args.zero),('A',args.arm_a),('B',args.arm_b)):
        require((path/'readback.json').is_file(),'evaluation must be read back first')
        require(read(path/'readback.json')['bank_sha256']==args.bank_sha256,'evaluation receipt drift')
        frozen[name],rows=evaluation_readback(path,bank,args.bank_sha256,cases,args.selection_sha256)
        conditional[name]=conditional_metrics(bank,rows)
    # Compare each endpoint to zero; do not treat A then B as a training trajectory.
    metrics={name:fit.evaluate_versions(bank['images'],dict(zero=frozen['zero'],**{name:frozen[name]})) for name in ('A','B')}
    write(args.output,dict(regime=bank['regime'],bank_sha256=args.bank_sha256,ordinary=metrics,conditional=conditional,
        limitations='CPU checks do not establish native parity or efficacy; unmatched predictions are annotation-relative unknown.'))
    return dict(status='offline_complete',output=str(args.output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare','readback','run','evaluate','arm-readback','eval-readback','offline'))
    parser.add_argument('--bank', type=Path, required=True)
    parser.add_argument('--bank-sha256',required=True)
    parser.add_argument('--selection',type=Path)
    parser.add_argument('--selection-sha256')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--arm',choices=('A','B'))
    parser.add_argument('--updates',type=int,default=16)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--release',type=Path)
    parser.add_argument('--release-sha256')
    parser.add_argument('--zero',type=Path)
    parser.add_argument('--arm-a',type=Path)
    parser.add_argument('--arm-b',type=Path)
    args = parser.parse_args()
    if args.command == 'prepare':
        print(json.dumps(prepare_packet(args)));return
    elif args.command == 'readback':
        bank = verify_bank(args.bank, args.bank_sha256)
    elif args.command in ('run','evaluate'):
        require(args.release is not None and args.release_sha256 is not None and args.output is not None,'exact lead release required')
        (run_arm if args.command=='run' else evaluate)(args)
        return
    elif args.command=='arm-readback':
        print(json.dumps(arm_readback(args.output,verify_bank(args.bank,args.bank_sha256),args.bank_sha256,args.arm,args.updates)))
        return
    elif args.command=='eval-readback':
        bank=verify_bank(args.bank,args.bank_sha256)
        cases=selection_cases(args.selection,args.selection_sha256,bank,args.bank_sha256)
        records,conditional=evaluation_readback(args.output,bank,args.bank_sha256,cases,args.selection_sha256)
        value=dict(status='complete',bank_sha256=args.bank_sha256,ordinary_requests=len(records),conditional_requests=len(conditional),frozen_sha256=fit.sha(args.output/'frozen.json'))
        write(args.output/'readback.json',value);print(json.dumps(value));return
    else:
        print(json.dumps(offline(args)));return
    print(json.dumps(dict(status='CPU_prepared', bank=str(args.bank), sha256=fit.sha(args.bank),
        counts=bank['counts'], jobs=len(bank['jobs']), classes=bank['classes'], conditional_counts=bank['conditional_counts'])))
