"""One screened-teacher fit: sealed CPU inputs and cold native endpoint.

Training is the already-qualified training.py; this lane adds no loss or scheduler.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import signal
import time
import traceback

from probes.dora_owner_learning.candidate_opportunity import digest, file_hash, require
from probes.parallel_owner_research.history import binding, publish, read, continuation_ledger

BASE = Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration')
ROOT = BASE/'2026-09-12-parallel-owner-research/data-flywheel'
HISTORY_INPUT = BASE/'2026-09-12-parallel-owner-research/history/preparation/training-inputs.json'
ENDPOINT_SOURCE = BASE/'2026-09-11-positive-progress-matched-control/endpoint-preparation/packet.json'
EXAMPLE = 'coco2017_train_000000417044'
ARM = 'screened25'
CAP, EOS = 3084, 151645
ENDPOINT_GPUS = (0, 1, 2, 3, 4, 5, 6, 7)
KEEPS = ['P0','P1','P2','P6','P7','P9','P10','P11','P12','P14','P15','P17','P18','P19','P21','P22','P24','P26','P27','P28','P29','P30','P31','P32','P33']


def prepare():
    from probes.parallel_owner_research.training import prepare_packet, work_counts
    records = read(ROOT/'proposed-literal-positive-records.json')
    sidecar = read(ROOT/'candidate-sidecar.json')
    reference = read(HISTORY_INPUT)
    require([r['source_row_id'] for r in records] == KEEPS, 'root-admitted 25 rows/order')
    clean = []
    for r in records:
        require(r['prefix_token_ids'] == clean and EOS not in r['target_token_ids'], 'literal compacted history/no EOS')
        require(r['example_id']==EXAMPLE, 'single exposed training image')
        clean.extend(r['target_token_ids'])
    require(len(clean)==249 and EXAMPLE not in reference['normal_keys'], '249 tokens/disjoint normal56')
    require(len(reference['normal_keys'])==56, 'fixed normal56')
    require([r['row_id'] for r in sidecar['rows'] if r['proposed_action']=='keep']==KEEPS, 'admitted sidecar identity')
    source = read(ENDPOINT_SOURCE)
    require(len(source['eval_records'])==384 and len({r['example_id'] for r in source['eval_records']})==384, 'legacy union384')
    sources = {name:binding(path) for name,path in {
        'sidecar':ROOT/'candidate-sidecar.json','records':ROOT/'proposed-literal-positive-records.json',
        'cleaned':ROOT/'cleaned-trajectory.json','endpoint_source':ENDPOINT_SOURCE,
        'history_input':HISTORY_INPUT,'producer':Path(__file__).resolve(),
        'native_ledger':Path(__file__).with_name('history.py'),
    }.items()}
    out = ROOT/'preparation-v2'
    runtime = {'world_sizes':[2], 'max_rank_seconds':10000,
               'max_cuda_allocated_bytes':32*1024**3,'max_cuda_reserved_bytes':32*1024**3,
               'max_rss_bytes':40*1024**3,'max_model_forwards_per_rank':1802,'max_image_forwards_per_rank':1802}
    train = prepare_packet(out/'training-inputs.json', lane='data-flywheel-screened25',
        anchor_input_path=reference['anchor_input']['path'],margin_input_path=reference['margin_input']['path'],
        normal_keys=reference['normal_keys'], positive_records=records,conditional_records=[],
        arms={ARM:{'steps':[[{'record_id':r['record_id'],'weight':1.0} for r in records] for _ in range(32)]}},
        weights={'positive':1.,'conditional_kl':0.,'normal_kl':100.,'margin':10.},
        denominators={'positive':25.,'conditional_kl':1.,'normal_kl':56.,'margin':56.},
        optimizer=reference['optimizer'],clip_gradient_norm=1.,runtime=runtime)
    counts = [work_counts(train,ARM,2,rank) for rank in (0,1)]
    require([r['model_forwards'] for r in counts]==[1802,1752], 'exact shared-engine forward plan')
    pilot = {'schema':'data_flywheel.pilot.v1','status':'sealed_prepared_no_gpu_launch',
             'sources':sources,'training_input':binding(out/'training-inputs.json'),
             'arm':ARM,'training_root':str(ROOT/'training-v1'),
             'normal_keys':reference['normal_keys'],'keep_row_ids':KEEPS,
             'training_counts_per_rank':counts,'cold_score_forwards':25,
             'endpoint_bounds':{'max_rank_seconds':12000,'max_memory_bytes':32*1024**3,
                                'max_model_forwards_per_rank':49*CAP,'max_image_forwards_per_rank':49},
             'endpoint_jobs_per_rank':[49,48,48,48,48,48,48,48], 'physical_gpus':list(ENDPOINT_GPUS),
             'outcome':{'reviewed_owner_iou':.5,'strict_duplicate_iou_exclusive':.95,
                        'strong_required_reviewed_owners':25,'eos_required':True,'invalid_or_malformed_allowed':0,
                        'physical_duplicate_or_group_extent_requires_visual_review':True,
                        'partitions':{'trained':1,'normal56':56,'other327':327},
                        'no_fresh_transfer':True,'no_eos_loss':True,'fixed_updates':32}}
    publish(out/'pilot.json',pilot)
    return validate(out/'pilot.json')


def validate(path):
    from probes.parallel_owner_research.training import load_packet, work_counts
    packet=read(path)
    require(packet['schema']=='data_flywheel.pilot.v1' and packet['arm']==ARM, 'pilot schema/arm')
    for value in [*packet['sources'].values(),packet['training_input']]:
        require(file_hash(value['path'])==value['sha256'],f"changed sealed source {value['path']}")
    train,_,_,_=load_packet(Path(packet['training_input']['path']))
    require(train['weights']=={'positive':1.,'conditional_kl':0.,'normal_kl':100.,'margin':10.}, 'fixed weights')
    require(train['denominators']=={'positive':25.,'conditional_kl':1.,'normal_kl':56.,'margin':56.}, 'fixed denominators')
    require(not train['conditional_records'] and len(train['arms'][ARM]['steps'])==32, 'no conditional/32 updates')
    require(set(train['arms'])=={ARM} and train['normal_keys']==packet['normal_keys'], 'single arm/normal identity')
    require([r['source_row_id'] for r in train['positive_records']]==KEEPS, 'fixed positive IDs')
    require([work_counts(train,ARM,2,s) for s in (0,1)]==packet['training_counts_per_rank'], 'counter contract')
    source=read(packet['sources']['endpoint_source']['path'])
    ids={r['example_id'] for r in source['eval_records']}
    require(len(ids)==384 and EXAMPLE in ids and set(packet['normal_keys'])<=ids and EXAMPLE not in packet['normal_keys'], 'legacy partitions')
    require(packet['physical_gpus']==list(ENDPOINT_GPUS) and packet['endpoint_jobs_per_rank']==[len(endpoint_jobs(source,s)) for s in range(8)],'fixed eight-shard placement')
    return packet


def prepare_endpoint(pilot_path, output):
    from probes.parallel_owner_research.training import verify_receipt
    from src.adapters.dora import inspect_dora_adapter_payload
    packet=validate(pilot_path)
    root=Path(packet['training_root'])
    receipt=verify_receipt(root)
    require(receipt['input']==packet['training_input'] and receipt['arm']==ARM and receipt['updates']==32, 'fixed training endpoint')
    cold=read(root/'cold-check.json')
    require(cold['status']=='passed' and cold['schema']=='parallel_owner_training.cold_check.v1', 'cold reload gate')
    require(cold['training_receipt']==binding(root/'receipt.json') and cold['saved_adapter']==receipt['saved_adapter'], 'cold/receipt identity')
    source=read(packet['sources']['endpoint_source']['path'])
    sidecar=read(packet['sources']['sidecar']['path'])
    stable=inspect_dora_adapter_payload(sidecar['source']['checkpoint'],source['model']['base_model_path'])
    ep={'schema':'data_flywheel.endpoint.v1','pilot':binding(pilot_path),
        'training_receipt':binding(root/'receipt.json'),'cold_check':binding(root/'cold-check.json'),
        'adapters':{'Stable50':stable,ARM:receipt['saved_adapter']},'bounds':packet['endpoint_bounds']}
    publish(output,ep)
    return ep


def endpoint_jobs(source,shard):
    require(shard in range(len(ENDPOINT_GPUS)),'eight fixed endpoint shards')
    jobs=[{'arm':ARM,'example_id':r['example_id']} for r in source['eval_records']][shard::len(ENDPOINT_GPUS)]
    return ([{'arm':'Stable50','example_id':EXAMPLE}] if shard==0 else [])+jobs


def reviewed_score(parsed,sidecar):
    from probes.dora_owner_learning.reward_rows import _pred_objects
    from src.eval.assignment import global_matches
    kept=[r for r in sidecar['rows'] if r['proposed_action']=='keep']
    gt=[(r['description'],tuple(r['bbox_xyxy_pixels'])) for r in kept]
    preds,invalid=_pred_objects(parsed)
    matches=global_matches(gt,preds,.5)
    owners=[kept[i]['row_id'] for i,_,_ in matches]
    partitions={'annotated':[r['row_id'] for r in kept if r['candidate_label']=='gt_matched'],
                'supplied_source_unlabeled':['P1'],
                'free_discovery_unlabeled':[r['row_id'] for r in kept if r['candidate_label']=='trusted_unlabeled_single_owner' and r['row_id']!='P1']}
    return {'tp':len(matches),'owners':owners,'unrecovered':[r['row_id'] for r in kept if r['row_id'] not in owners],
            'matches':[{'owner_row_id':kept[i]['row_id'],'pred_index':j,'iou':v} for i,j,v in matches],
            'partitions':{k:{'recovered':sorted(set(v)&set(owners)),'denominator':len(v)} for k,v in partitions.items()},
            'prediction_count':len(preds),'invalid':invalid,
            'scope':'candidate-owner geometric matching; unmatched predictions are not hallucinations'}


def endpoint_rank(endpoint_path,output,shard,physical_gpu):
    import torch
    from probes.dora_owner_learning.runtime import load_policy
    from probes.dora_owner_learning.route_access import checkpoint_config
    from probes.source_rweak_row_cross.run import build_requests
    from src.adapters.dora import inspect_dora_adapter_payload
    from src.config.inference import InferConfig
    from src.qwen.generation import NativeGenerationPolicy,generate_continuations
    from src.qwen.native import prepare_native_inputs
    ep=read(endpoint_path)
    require(ep['schema']=='data_flywheel.endpoint.v1','endpoint schema')
    for bound in (ep['pilot'],ep['training_receipt'],ep['cold_check']):
        require(file_hash(bound['path'])==bound['sha256'],'endpoint dependency changed')
    packet=validate(ep['pilot']['path'])
    source=read(packet['sources']['endpoint_source']['path'])
    by_id={r['example_id']:r for r in source['eval_records']}
    require(shard in range(8) and ENDPOINT_GPUS[shard]==physical_gpu,'pilot GPU reservation')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')==str(physical_gpu) and torch.cuda.device_count()==1,'one assigned GPU')
    output=Path(output)
    require(not output.exists(),'endpoint output collision')
    output.mkdir(parents=True)
    jobs=endpoint_jobs(source,shard)
    started=time.monotonic()
    terminal={'status':'running','shard':shard,'physical_gpu':physical_gpu,'pid':os.getpid(),
              'model_loads':0,'model_forwards':0,'image_forwards':0,'continuations':0,'new_tokens':0,
              'endpoint_packet':binding(endpoint_path)}
    publish(output/'launch.json',terminal)
    old_alarm=signal.getsignal(signal.SIGALRM)
    handles=[]
    try:
        signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('endpoint wall bound')))
        signal.alarm(ep['bounds']['max_rank_seconds'])
        torch.cuda.reset_peak_memory_stats()
        policy=NativeGenerationPolicy(temperature=0.,top_p=1.,top_k=0,repetition_penalty=1.,use_model_defaults=False)
        current_arm=None
        qwen=None
        with (output/'rows.jsonl').open('x') as stream:
            for job in jobs:
                if job['arm']!=current_arm:
                    for h in handles:h.remove()
                    handles=[]
                    if qwen is not None:
                        del qwen
                        import gc
                        gc.collect();torch.cuda.empty_cache()
                    adapter=ep['adapters'][job['arm']]
                    require(inspect_dora_adapter_payload(adapter['root'],source['model']['base_model_path'])==adapter,'adapter identity')
                    config=checkpoint_config(InferConfig.model_validate(source['config']),adapter['root'])
                    qwen,identity=load_policy(config,device=torch.device('cuda:0'))
                    terminal['model_loads']+=1
                    live=identity['model_identity']['adapter'];observed=identity['effective_settings']
                    require(live['adapter_path']==adapter['root'] and live['merged_adapters']==[],'unmerged exact adapter')
                    require(observed['observed_model_dtype']['parameter_dtype_names']==['torch.float32'] and observed['observed_attn_implementation']=='sdpa','FP32 SDPA')
                    publish(output/f"model-{job['arm']}.json",identity)
                    qwen.model.eval()
                    for p in qwen.model.parameters():p.requires_grad_(False)
                    def model_hook(*_):
                        require(terminal['model_forwards']<ep['bounds']['max_model_forwards_per_rank'],'model forward cap')
                        terminal['model_forwards']+=1
                    def image_hook(*_):
                        require(terminal['image_forwards']<ep['bounds']['max_image_forwards_per_rank'],'image forward cap')
                        terminal['image_forwards']+=1
                    visuals=[m for n,m in qwen.model.named_modules() if n.endswith('visual')]
                    require(len(visuals)==1,'one visual module')
                    handles=[qwen.model.register_forward_pre_hook(model_hook),visuals[0].register_forward_pre_hook(image_hook)]
                    current_arm=job['arm']
                frozen=by_id[job['example_id']]
                requests,_=build_requests(qwen,source['config'],[frozen['case']])
                batch=prepare_native_inputs(qwen.processor,requests,device=torch.device('cuda:0'),record_media_identity=True)
                plan=frozen['case']['image_plan']
                require(list(batch.prompt_token_ids[0])==frozen['prompt_token_ids'] and batch.media_sha256[0]==plan['executed_media_sha256'] and list(batch.image_grids[0])==plan['observed_image_grid_thw'],'native prompt/image/grid identity')
                with torch.inference_mode():
                    generated=generate_continuations(qwen.model,batch,extensions=[[]],budgets=[CAP],eos_token_id=EOS,pad_token_id=qwen.tokenizer.pad_token_id,policy=policy,trace='none')[0]
                ids=list(generated.token_ids)
                require(generated.request_id==job['example_id'] and 0<len(ids)<=CAP,'request/token count')
                require((generated.stop_reason=='im_end' and ids[-1]==EOS and EOS not in ids[:-1]) or (generated.stop_reason=='length' and len(ids)==CAP and EOS not in ids),'EOS/cap identity')
                text=qwen.tokenizer.decode(ids,skip_special_tokens=False)
                row={**job,'shard':shard,'endpoint_packet_sha256':file_hash(endpoint_path),
                     'free_token_ids':ids,'free_token_ids_sha256':digest(ids),'free_text':text,'stop_reason':generated.stop_reason,
                     'prompt_token_ids_sha256':digest(frozen['prompt_token_ids']),'executed_media_sha256':batch.media_sha256[0],
                     'prefix_token_ids':[],**continuation_ledger('',text,frozen,0,len(ids),generated.stop_reason)}
                stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush();os.fsync(stream.fileno())
                terminal['continuations']+=1;terminal['new_tokens']+=len(ids)
        require(terminal['continuations']==len(jobs),'all endpoint jobs')
        terminal.update(status='completed',exit_code=0)
    except BaseException as exc:
        terminal.update(status='failed',exit_code=1,error=repr(exc),traceback=traceback.format_exc());raise
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,old_alarm)
        for h in handles:h.remove()
        terminal.update(elapsed_seconds=time.monotonic()-started,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0,
                        peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0,
                        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        if terminal['status']=='completed' and any(terminal[k]>ep['bounds']['max_memory_bytes'] for k in ('peak_cuda_allocated_bytes','peak_cuda_reserved_bytes','peak_rss_bytes')):
            terminal.update(status='failed',exit_code=1,error='endpoint memory bound')
        publish(output/'terminal.json',terminal)
    require(terminal['status']=='completed','endpoint completion')


def reduce_endpoint(endpoint_path,output):
    from transformers import AutoTokenizer
    ep=read(endpoint_path);packet=validate(ep['pilot']['path']);output=Path(output)
    source=read(packet['sources']['endpoint_source']['path']);by_id={r['example_id']:r for r in source['eval_records']}
    sidecar=read(packet['sources']['sidecar']['path'])
    tokenizer=AutoTokenizer.from_pretrained(source['model']['base_model_path'],local_files_only=True)
    rows=[];terminals=[]
    for shard in range(8):
        terminal=read(output/f'shard-{shard}/terminal.json')
        require(terminal['status']=='completed' and terminal['exit_code']==0 and terminal['endpoint_packet']==binding(endpoint_path),'endpoint terminal')
        require((output/f'shard-{shard}.exit').read_text().strip()=='0','actual outer process exit')
        actual=[json.loads(x) for x in (output/f'shard-{shard}/rows.jsonl').read_text().splitlines() if x.strip()]
        jobs=endpoint_jobs(source,shard)
        require(len(actual)==len(jobs)==terminal['continuations'],'endpoint denominator')
        for r,j in zip(actual,jobs):
            require(all(r[k]==v for k,v in j.items()) and r['shard']==shard and r['prefix_token_ids']==[],'cold native job identity')
            require(r['endpoint_packet_sha256']==file_hash(endpoint_path),'row packet identity')
            require(digest(r['free_token_ids'])==r['free_token_ids_sha256'] and tokenizer.decode(r['free_token_ids'],skip_special_tokens=False)==r['free_text'],'raw token/text')
            f=by_id[r['example_id']]
            require(r['prompt_token_ids_sha256']==digest(f['prompt_token_ids']) and r['executed_media_sha256']==f['case']['image_plan']['executed_media_sha256'],'cold media identity')
            fresh=continuation_ledger('',r['free_text'],f,0,len(r['free_token_ids']),r['stop_reason'])
            require(all(r[k]==v for k,v in fresh.items()),'fresh owner/parser/burden reduction')
        rows.extend(actual);terminals.append(terminal)
    trained=[r for r in rows if r['arm']==ARM];baseline=next(r for r in rows if r['arm']=='Stable50')
    require(len(trained)==384 and len(rows)==385,'384+1 endpoint population')
    target=next(r for r in trained if r['example_id']==EXAMPLE)
    old=reviewed_score(baseline['free_parsed'],sidecar);new=reviewed_score(target['free_parsed'],sidecar)
    partitions={'trained':{EXAMPLE},'normal56':set(packet['normal_keys'])}
    partitions['other327']=set(by_id)-partitions['trained']-partitions['normal56']
    require([len(partitions[k]) for k in ('trained','normal56','other327')]==[1,56,327],'disjoint legacy partitions')
    panels={}
    for name,ids in partitions.items():
        subset=[r for r in trained if r['example_id'] in ids]
        panel={'images':len(subset),'owners':{},'burden':{}}
        for t in ('50','60','80'):
            entries=[]
            for r in subset:
                a=by_id[r['example_id']]['stable_score'][t];b=r['free_score'][t]
                entries.append({'example_id':r['example_id'],'gained':sorted(set(b['owners'])-set(a['owners'])),'lost':sorted(set(a['owners'])-set(b['owners'])),
                                'retained':sorted(set(a['owners'])&set(b['owners'])),'baseline':a,'trained':b})
            panel['owners'][t]={'gained_count':sum(len(x['gained']) for x in entries),'lost_count':sum(len(x['lost']) for x in entries),
                'baseline':{k:sum(x['baseline'][k] for x in entries) for k in ('tp','fp','fn')},
                'trained':{k:sum(x['trained'][k] for x in entries) for k in ('tp','fp','fn')},'per_image':entries}
            for arm in ('baseline','trained'):
                counts=panel['owners'][t][arm]
                counts['f1']=2*counts['tp']/(2*counts['tp']+counts['fp']+counts['fn']) if counts['tp'] else 0.
        for key in target['burden']:
            panel['burden'][key]=sum(r['burden'][key] for r in subset)
        for key in ('strict_repeats','parser_drops','invalid_predictions','prediction_count','cap','complete_token_length'):
            panel.setdefault('burden_comparison',{})[key]={'baseline':sum(by_id[r['example_id']]['stable_score'][key] for r in subset),
                                                         'trained':sum(r['free_score'][key] for r in subset)}
        old_drops=[d for r in subset for d in by_id[r['example_id']]['stable_parsed']['dropped_predictions']]
        old_invalid=sum(d.get('reason')=='geometry_invalid' for d in old_drops)
        panel['baseline_burden']={'free_row_starts':sum(by_id[r['example_id']]['stable_parsed']['raw_decode_text'].count('<|object_ref_start|>') for r in subset),
            'free_valid_rows':sum(len(by_id[r['example_id']]['stable_parsed']['pred']) for r in subset),
            'free_strict_repeats_including_history':sum(by_id[r['example_id']]['stable_score']['strict_repeats'] for r in subset),
            'free_geometry_invalid':old_invalid,'free_other_malformed':len(old_drops)-old_invalid,'supplied_row_starts':0,
            'cap':sum(by_id[r['example_id']]['stable_score']['cap'] for r in subset),
            'eos':sum(1-by_id[r['example_id']]['stable_score']['cap'] for r in subset)}
        panels[name]=panel
    machine_pass=(new['tp']==25 and set(old['owners'])<=set(new['owners']) and target['stop_reason']=='im_end' and target['free_parsed']['dropped_prediction_count']==0 and new['invalid']==0)
    result={'schema':'data_flywheel.result.v1','status':'candidate_cold_verified_visual_review_pending','endpoint_packet':binding(endpoint_path),
            'reviewed':{'Stable50':old,ARM:new,'gained':sorted(set(new['owners'])-set(old['owners'])),'lost':sorted(set(old['owners'])-set(new['owners'])),
                        'machine_strong_criterion':machine_pass,'physical_duplicate_group_extent_visual_gate':'pending'},
            'legacy_panels':panels,'native_baseline_matches_retained_tokens':baseline['free_token_ids']==by_id[EXAMPLE]['stable_ids'],
            'cost':{'model_forwards':sum(t['model_forwards'] for t in terminals),'image_forwards':sum(t['image_forwards'] for t in terminals),
                    'allocated_gpu_hours':sum(t['elapsed_seconds'] for t in terminals)/3600},
            'row_sources':[binding(output/f'shard-{s}/rows.jsonl') for s in range(8)]}
    for r in (baseline,target):
        directory=output/'visual-inputs'/r['arm'];directory.mkdir(parents=True)
        text=json.dumps(r['free_parsed'],ensure_ascii=False)+'\n'
        for name in ('gt_vs_pred.jsonl','gt_vs_pred_scored.jsonl'):
            with (directory/name).open('x') as f:f.write(text)
    publish(output/'result.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','verify','prepare-endpoint','endpoint-rank','reduce'])
    p.add_argument('--pilot',type=Path,default=ROOT/'preparation-v2/pilot.json')
    p.add_argument('--endpoint',type=Path,default=ROOT/'endpoint-v1.json')
    p.add_argument('--output',type=Path,default=ROOT/'endpoint-v1')
    p.add_argument('--shard',type=int,choices=list(range(8)));p.add_argument('--physical-gpu',type=int,choices=list(ENDPOINT_GPUS))
    a=p.parse_args()
    if a.command=='prepare':result=prepare()
    elif a.command=='verify':result=validate(a.pilot)
    elif a.command=='prepare-endpoint':result=prepare_endpoint(a.pilot,a.endpoint)
    elif a.command=='endpoint-rank':
        require(a.shard is not None and a.physical_gpu is not None,'assigned endpoint shard/GPU')
        endpoint_rank(a.endpoint,a.output,a.shard,a.physical_gpu);return
    else:result=reduce_endpoint(a.endpoint,a.output)
    print(json.dumps({'status':result.get('status','prepared'),'schema':result['schema']}))


if __name__=='__main__':main()
