"""Fixed pilot cache, paired fits, and independently reloaded evaluation."""
from __future__ import annotations
import argparse
from functools import partial
import hashlib
import json
from pathlib import Path
import time
import torch
from .runtime import RunLedger, write_once, replay
from .qualify import grammar, frozen_digest, build_batch, target_tokens, prediction
from .bridge import CoordinateAddressReadout, coordinate_role_from_prefix
from .training import batch_backward
from probes.training_set_completion.artifacts import binding
from probes.training_set_completion.untied_shared import load_model


def state_hash(module):
    h = hashlib.sha256()
    for name, value in module.state_dict().items():
        h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def device_cache(cache, device):
    return {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in cache.items()}


def run(args):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest = json.loads(args.manifest.read_text())
    for entry in manifest['source_bindings']:
        assert binding(Path(entry['path'])) == entry, 'production input/source drift'
    ledger = RunLedger(args.output,args.device,production=True)
    try:
        ledger.check_budget()
        write_once(args.output/'manifest-binding.json',binding(args.manifest))
        if args.mode == 'train':
            train(args,manifest,ledger)
        else:
            q,identity = load_model('tied',torch.device(args.device))
            q.processor.image_processor.do_resize = False
            ledger.capture_sources(); ledger.attach(q.model)
            before = frozen_digest(q.model)
            write_once(args.output/'source.json',dict(identity=identity,runtime=q.to_artifact_dict(),frozen_sha256=before))
            if args.mode == 'cache':
                cache_source(args,manifest,ledger,q,before)
            else:
                evaluate(args,manifest,ledger,q)
            assert frozen_digest(q.model) == before, 'frozen backbone changed'
        ledger.finish('complete')
    except BaseException as exc:
        ledger.finish('HOLD',repr(exc))
        raise


def case_config(manifest,case):
    return case.get('model_config',manifest['model_config'])


def cache_source(args,manifest,ledger,q,source_hash):
    g=grammar(q.tokenizer); ids=g['coordinate_token_ids']; bins={t:i for i,t in enumerate(ids)}
    parser=partial(coordinate_role_from_prefix,**g)
    entries={}
    for case in manifest['training_cases']+manifest['calibration_cases']:
        ledger.check_budget(); config=case_config(manifest,case)
        batch=build_batch(q,case,config,torch.device(args.device))
        tokens=target_tokens(q,case,config,row_limit=None)
        positions=[i for i,t in enumerate(tokens) if t in bins]
        roles=[parser(tokens[:i]) for i in positions]
        assert all(r>=0 for r in roles)
        start=time.time(); logits,hidden,visual=replay(q.model,batch,tokens)
        selected=logits[positions]
        cache=dict(coordinate_logits=selected[:,ids].detach().cpu(),full_lse=selected.logsumexp(-1).detach().cpu(),
            hidden=hidden[positions].detach().cpu(),visual=visual.detach().cpu(),roles=torch.tensor(roles),
            targets=torch.tensor([bins[tokens[i]] for i in positions]),grid=tuple(int(x)//2 for x in batch.image_grids[0][1:]),
            coordinate_ids=ids,source_hash=source_hash,history=tokens,prompt=list(batch.prompt_token_ids[0]),row_id=case['row_id'])
        path=args.output/f"cache-{case['row_id']}.pt"; torch.save(cache,path)
        entries[str(case['row_id'])]=dict(binding=binding(path),coordinate_tokens=len(positions),seconds=time.time()-start,
            media_sha256=list(batch.media_sha256),grid=list(batch.image_grids[0]))
        del logits,hidden,visual,selected,cache
    write_once(args.output/'index.json',dict(entries=entries,source_hash=source_hash,manifest=binding(args.manifest)))


def train(args,manifest,ledger):
    index=json.loads((args.cache/'index.json').read_text()); entries=index['entries']
    for entry in entries.values():
        assert binding(Path(entry['binding']['path'])) == entry['binding'], 'cache drift'
    caches={row:torch.load(e['binding']['path'],map_location='cpu',weights_only=True) for row,e in entries.items()}
    first=next(iter(caches.values())); torch.manual_seed(args.seed)
    sidecar=CoordinateAddressReadout(first['coordinate_ids'],permuted=args.arm=='permuted').to(args.device)
    initial=state_hash(sidecar)
    optimizer=torch.optim.AdamW(sidecar.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=0)
    assert len(optimizer.param_groups)==1
    assert {id(p) for p in optimizer.param_groups[0]['params']} == {id(p) for p in sidecar.parameters()}
    ledger.capture_sources()
    write_once(args.output/'initial.json',dict(seed=args.seed,arm=args.arm,sha256=initial,parameters=sum(p.numel() for p in sidecar.parameters()),
        optimizer_allowlist=[n for n,_ in sidecar.named_parameters()],cache_index=binding(args.cache/'index.json')))
    schedule=manifest['schedules'][str(args.seed)]
    assert len(schedule)==256 and all(len(b)==8 for b in schedule)
    with (args.output/'learning.jsonl').open('x') as stream:
        for update,rows in enumerate(schedule,1):
            ledger.check_budget(); start=time.time()
            batch=[device_cache(caches[str(row)],args.device) for row in rows]
            report=batch_backward(sidecar,batch,check=update<=2)
            if update==1:
                assert report['gradient_norms']['gain']>0
                assert all(v==0 for k,v in report['gradient_norms'].items() if k!='gain')
            if update==2:
                assert all(v>0 for v in report['gradient_norms'].values()), 'dead sidecar branch'
            optimizer.step()
            report.update(update=update,row_ids=rows,seconds=time.time()-start,gain=float(sidecar.gain.detach()))
            stream.write(json.dumps(report,allow_nan=False)+'\n'); stream.flush()
            if update<=2: write_once(args.output/f'batch-check-{update}.json',report)
            del batch
    saved=dict(state_dict={k:v.detach().cpu() for k,v in sidecar.state_dict().items()},seed=args.seed,arm=args.arm,
        update=256,initial_sha256=initial,final_sha256=state_hash(sidecar),coordinate_ids=first['coordinate_ids'],
        source_hash=index['source_hash'],manifest=binding(args.manifest),cache_index=binding(args.cache/'index.json'))
    torch.save(saved,args.output/'update-256.pt')
    write_once(args.output/'checkpoint.json',binding(args.output/'update-256.pt'))


def evaluate(args,manifest,ledger,q):
    g=grammar(q.tokenizer); parser=partial(coordinate_role_from_prefix,**g)
    bridges={}; checkpoints={}
    source_hash=frozen_digest(q.model)
    for seed in (1729,2718):
        for arm in ('aligned','permuted'):
            name=f'{arm}-{seed}'
            path=args.fits/name/'update-256.pt'
            if not path.exists():
                continue
            saved=torch.load(path,map_location='cpu',weights_only=True)
            assert saved['manifest']==binding(args.manifest) and saved['update']==256
            assert saved['arm']==arm and saved['seed']==seed
            assert saved['coordinate_ids']==g['coordinate_token_ids']
            assert saved['source_hash']==source_hash
            sidecar=CoordinateAddressReadout(g['coordinate_token_ids'],permuted=arm=='permuted').to(args.device)
            sidecar.load_state_dict(saved['state_dict']); sidecar.eval()
            assert state_hash(sidecar)==saved['final_sha256']
            bridges[name]=sidecar; checkpoints[name]=binding(path)
    write_once(args.output/'reload.json',dict(checkpoints=checkpoints,independent_process=True))
    cache_index=json.loads((args.cache/'index.json').read_text())
    cases=[('native',c) for c in manifest['native_cases']]+[('calibration',c) for c in manifest['calibration_cases']]
    for index,(kind,case) in enumerate(cases):
        if index % args.workers != args.worker: continue
        ledger.check_budget()
        batch=build_batch(q,case,case_config(manifest,case),torch.device(args.device))
        cache=None
        if kind=='calibration':
            entry=cache_index['entries'][str(case['row_id'])]['binding']
            assert binding(Path(entry['path']))==entry
            cache=device_cache(torch.load(entry['path'],map_location='cpu',weights_only=True),args.device)
        for condition in manifest['conditions']:
            name=condition if isinstance(condition,str) else condition['name']
            if name!='original' and name not in bridges: continue
            ledger.check_budget(); started=time.time()
            sidecar=bridges.get(name)
            cell=dict(kind=kind,row_id=case['row_id'],condition=name,cohort=case['cohort'],status='complete',
                checkpoint=checkpoints.get(name),case_binding=dict(manifest=binding(args.manifest),row_id=case['row_id']),
                prompt_token_ids=list(batch.prompt_token_ids[0]),media_sha256=list(batch.media_sha256))
            path=args.output/'cells'/f"{kind}-{case['row_id']}-{name}.json"
            try:
                if kind=='native':
                    result=prediction(q,batch,case,sidecar,parser,cap=3084)
                    result.pop('saved_logits',None);cell.update(result)
                else:
                    with torch.no_grad():
                        z=cache['coordinate_logits'] if sidecar is None else sidecar.adjust_coordinate_logits(
                            cache['coordinate_logits'],cache['hidden'],cache['visual'],cache['roles'],*cache['grid'])
                        ce=cache['full_lse']-z.gather(1,cache['targets'][:,None]).squeeze(1)
                        cell.update(teacher_coordinate_ce_sum=float(ce.sum()),coordinate_tokens=len(ce),
                            coordinate_absolute_error_sum=float((z.argmax(-1)-cache['targets']).abs().sum())/1000)
                    referent=manifest['calibration_referents'][str(case['row_id'])]
                    extension=q.tokenizer.encode('<|object_ref_start|>'+referent+'<|object_ref_end|><|box_start|>',add_special_tokens=False)
                    result=prediction(q,batch,case,sidecar,parser,extension=extension,cap=8)
                    result.pop('saved_logits',None);cell.update(free=result,description=referent,description_only_prefix=extension)
                cell['seconds']=time.time()-started; write_once(path,cell)
            except BaseException as exc:
                cell.update(status='HOLD',error=repr(exc),seconds=time.time()-started)
                if hasattr(exc,'partial_generation'): cell.update(partial_generation=exc.partial_generation)
                write_once(path,cell)
                raise



def main():
    p=argparse.ArgumentParser(); p.add_argument('--mode',choices=['cache','train','evaluate'],required=True)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',required=True);p.add_argument('--cache',type=Path)
    p.add_argument('--seed',type=int);p.add_argument('--arm',choices=['aligned','permuted'])
    p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=8)
    p.add_argument('--fits',type=Path)
    run(p.parse_args())

if __name__=='__main__': main()
