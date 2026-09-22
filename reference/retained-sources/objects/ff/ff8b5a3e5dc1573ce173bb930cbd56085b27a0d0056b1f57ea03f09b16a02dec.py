"""Locked KL10 dev112: four independent native-greedy shards, no new training."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import resource
import signal
import time
import traceback

from probes.dora_owner_learning.candidate_opportunity import digest,file_hash,indexed,require,rows,score
from probes.dora_owner_learning.entrance_ce_eval import DEV,consume,validate_natural,aggregate_scores
from probes.dora_owner_learning.selective_preservation_eval import ARM_ROOT,validate_receipt
from probes.dora_owner_learning.route_access import CONFIG,checkpoint_config,checked_ids,publish
from probes.dora_owner_learning.reward_rows import _gt_objects
from probes.source_rweak_row_cross.owner_row_robustness import native_record
from probes.source_rweak_row_cross.run import build_requests

OUTPUT=Path(__file__).resolve().parent
GPUS=(1,3,4,5)
RECEIPT_SHA='99be264f3b19a7ab84722d4b36832490c0efa7b3f5225ff4941241903a6d0607'
TENSOR_SHA='712815e80d1323e74deda1b0aa94a8aeb1a6041c79d7ad9068819931f4b33002'


def partition(dev_ids,guard_ids,train_ids):
    dev=list(map(int,dev_ids));guard=list(map(int,guard_ids));train=set(map(int,train_ids))
    require(len(dev)==len(set(dev))==128 and len(guard)==len(set(guard))==16,'dev128/guard16 identity')
    require(set(guard)<=set(dev) and not train&set(dev),'guard/train population collision')
    selected=sorted(set(dev)-set(guard));shards=[selected[i::4] for i in range(4)]
    require(len(selected)==112 and all(len(s)==28 for s in shards),'locked112 partition')
    return selected,shards


def locked_checkpoint():
    path=ARM_ROOT/'training/receipt.json';require(file_hash(path)==RECEIPT_SHA,'locked receipt changed')
    receipt=json.loads(path.read_text());old=json.loads((ARM_ROOT/'evaluation/manifest.json').read_text())
    adapter=validate_receipt(receipt,old)
    require(file_hash(Path(adapter['root'])/'adapter_model.safetensors')==TENSOR_SHA,'locked adapter tensor changed')
    return receipt,adapter,old


def prepare():
    import yaml
    from src.data.geometry import parse_coord_token
    from src.qwen.runtime_loading import QwenLoadOptions,load_qwen_components_from_options
    from src.qwen.native import prepare_native_inputs
    require(not (OUTPUT/'manifest.json').exists(),'occupied manifest')
    receipt,adapter,old=locked_checkpoint()
    config=yaml.safe_load((DEV/'configs/resolved.yaml').read_text())['config']
    inputs=rows(config['data']['input_jsonl']);raw=indexed(rows(DEV/'gt_vs_pred.jsonl'),'row_id')
    images=indexed(rows(DEV/'image_plan.jsonl'),'row_id');manifest=json.loads((DEV/'run_manifest.json').read_text())
    prompts=indexed(manifest['prompt_trace'],'row_id')
    guard=[r['image_id'] for r in old['records'] if r['split']=='guard'];train=[r['image_id'] for r in old['records'] if r['split']=='train']
    selected,shards=partition([r['image_id'] for r in inputs],guard,train)
    selection=dict(image_ids=selected,guard_ids=guard,train_ids=train,shards=[dict(shard=i,gpu=GPUS[i],image_ids=s) for i,s in enumerate(shards)],
        rule='All historical Source dev128 minus exact guard16, numeric sorted-ID round robin; no outcome filtering/replacement.')
    publish(OUTPUT/'selection.json',selection)
    require(manifest['model_identity']==old['source_identity'],'Source checkpoint identity')
    policy=manifest['generation_policy'];require(not policy['do_sample'] and policy['max_new_tokens']==3084 and policy['repetition_penalty']==1 and policy['top_p']==1,'Source native policy')
    require(config['model']['dtype']=='fp32' and config['backend']['hf']==dict(attn_implementation='sdpa',patch_embed_linearization='enabled'),'Source numerics')
    traces=defaultdict(list)
    for t in rows(DEV/'pred_token_trace.jsonl'):
        if t['trace_type']=='generated_token' and not t['is_pad']:traces[t['row_id']].append(t)
    qwen=load_qwen_components_from_options(QwenLoadOptions(base_model=config['model']['base_model'],dtype='fp32',
        attn_implementation='sdpa',patch_embed_linearization='enabled',load_model=False))
    by_id={int(r['image_id']):r for r in inputs};records=[]
    for iid in selected:
        inp=by_id[iid];eid=f'coco2017_train_{iid:012d}';golden=raw[eid];plan=images[eid]
        gt=[dict(description=o['desc'],bbox=[parse_coord_token(v,field='GT') for v in o['bbox_2d']],object_id=str(o['coco_ann_id'])) for o in inp['objects']]
        require(_gt_objects(dict(golden,gt=gt),row_id=eid)==_gt_objects(golden,row_id=eid) and [g['object_id'] for g in gt]==[str(g['object_id']) for g in golden['gt']],'raw/Source GT identity')
        require(file_hash(golden['image_path'])==plan['image_content_sha256'],'Source image bytes')
        case=dict(row_id=eid,row_index=golden['row_index'],image_path=golden['image_path'],image_width=golden['image_width'],
            image_height=golden['image_height'],input_record=inp,image_plan=plan)
        requests,_=build_requests(qwen,config,[case]);batch=prepare_native_inputs(qwen.processor,requests,device='cpu',record_media_identity=True)
        prompt=list(batch.prompt_token_ids[0]);require(digest(prompt)==prompts[eid]['backend_executed_prompt_token_ids_sha256'],'Source prompt identity')
        require(batch.media_sha256[0]==plan['executed_media_sha256'] and list(batch.image_grids[0])==plan['observed_image_grid_thw'],'Source media/grid identity')
        trace=sorted(traces[eid],key=lambda t:t['generated_step_index']);require([t['generated_step_index'] for t in trace]==list(range(len(trace))),'Source native trace coverage')
        ids=checked_ids([t['token_id'] for t in trace],golden['decode_stop_reason'])
        require(qwen.tokenizer.decode(ids,skip_special_tokens=False)==golden['raw_decode_text'] and native_record(golden['raw_decode_text'],case,golden,golden['decode_stop_reason'])==golden,'Source token/parser identity')
        records.append(dict(example_id=eid,image_id=iid,split='dev112',case=case,prompt_token_ids=prompt,baseline=golden,baseline_ids=ids,
            baseline_score=score(golden,seed=-1,length=len(ids),stop=golden['decode_stop_reason'])))
        del batch
    paths=[DEV/'run_manifest.json',DEV/'configs/resolved.yaml',DEV/'gt_vs_pred.jsonl',DEV/'image_plan.jsonl',DEV/'pred_token_trace.jsonl',Path(config['data']['input_jsonl']),ARM_ROOT/'evaluation/manifest.json',ARM_ROOT/'training/receipt.json']
    packet=dict(schema='selective_preservation.locked_dev112.v1',selection=selection,records=records,config=config,source_model=old['source_model'],
        training_receipt_sha256=RECEIPT_SHA,adapter_tensor_sha256=TENSOR_SHA,source_files={str(p):file_hash(p) for p in paths},producer_sha256=file_hash(__file__))
    publish(OUTPUT/'manifest.json',packet)
    print(json.dumps(dict(status='frozen',records=112,shards=[len(s) for s in shards],GPUs=GPUS,manifest_sha256=digest(packet))))


def validate_packet(packet):
    require(packet['schema']=='selective_preservation.locked_dev112.v1' and packet['training_receipt_sha256']==RECEIPT_SHA and packet['adapter_tensor_sha256']==TENSOR_SHA,'locked packet identity')
    s=packet['selection'];selected,shards=partition(s['image_ids']+s['guard_ids'],s['guard_ids'],s['train_ids'])
    require(selected==s['image_ids'] and [r['image_id'] for r in packet['records']]==selected,'locked record ordering')
    require(s['shards']==[dict(shard=i,gpu=GPUS[i],image_ids=sh) for i,sh in enumerate(shards)],'changed round-robin shard')
    indexed(packet['records'],'example_id')
    for p,sha in packet['source_files'].items():require(file_hash(p)==sha,'Source artifact changed')


def shard_records(packet,shard):
    require(type(shard) is int and 0<=shard<4,'invalid shard')
    ids=set(packet['selection']['shards'][shard]['image_ids']);r=[r for r in packet['records'] if r['image_id'] in ids]
    require(len(r)==28 and len({x['example_id'] for x in r})==28,'shard28 coverage')
    return r


def execute(shard):
    import torch
    from src.config.inference import load_research_infer_config
    from src.qwen.native import prepare_native_inputs
    from src.qwen.generation import generate_continuations
    from probes.dora_owner_learning.runtime import load_policy
    packet=json.loads((OUTPUT/'manifest.json').read_text());validate_packet(packet);records=shard_records(packet,shard)
    receipt,adapter,_=locked_checkpoint();require(os.environ.get('CUDA_VISIBLE_DEVICES')==str(GPUS[shard]) and torch.cuda.device_count()==1,'assigned single GPU only')
    run=OUTPUT/f'shard-{shard}';run.mkdir(exist_ok=False);started=time.monotonic()
    terminal=dict(status='running',pid=os.getpid(),shard=shard,gpu=GPUS[shard],manifest_sha256=digest(packet),model_loads=0,score_forwards=0,
        model_forwards=0,image_forwards=0,continuations=0,new_tokens=0,producer_sha256=file_hash(__file__))
    publish(run/'launch.json',terminal)
    def expired(*_):raise TimeoutError('3600 second shard invocation limit')
    signal.signal(signal.SIGALRM,expired);signal.alarm(3600)
    try:
        config=checkpoint_config(load_research_infer_config(CONFIG).config,adapter['root'])
        require(str(config.model.base_model)==packet['source_model']['base_model_path'] and str(config.embedding_delta.path)==packet['source_model']['source_embedding']['root'],'base/embedding identity')
        require(config.model.dtype=='fp32' and config.backend.hf.attn_implementation=='sdpa' and config.backend.hf.patch_embed_linearization=='enabled','configured native numerics')
        qwen,identity=load_policy(config,device=torch.device('cuda:0'));terminal['model_loads']=1
        require(identity['model_identity']['adapter']['adapter_path']==adapter['root'] and not identity['model_identity']['adapter']['merged_adapters'],'saved unmerged candidate identity')
        require(identity['effective_settings']['observed_model_dtype']['parameter_dtype_names']==['torch.float32'] and identity['effective_settings']['observed_attn_implementation']=='sdpa','observed native numerics')
        publish(run/'model.json',identity);publish(run/'config.json',config.model_dump(mode='json'))
        qwen.model.eval();torch.cuda.reset_peak_memory_stats()
        def count_model(*_):terminal['model_forwards']+=1
        def count_image(*_):terminal['image_forwards']+=1
        qwen.model.register_forward_pre_hook(count_model);visual=[m for n,m in qwen.model.named_modules() if n.endswith('visual')]
        require(len(visual)==1,'visual module identity');visual[0].register_forward_pre_hook(count_image)
        path=run/'rows.jsonl'
        with path.open('x') as stream:
            for frozen in records:
                request,_=build_requests(qwen,packet['config'],[frozen['case']])
                require(list(request[0].expected_token_ids)==frozen['prompt_token_ids'],'reconstructed prompt identity')
                batch=prepare_native_inputs(qwen.processor,request,device=torch.device('cuda:0'),record_media_identity=True)
                plan=frozen['case']['image_plan'];require(list(batch.prompt_token_ids[0])==frozen['prompt_token_ids'] and batch.media_sha256[0]==plan['executed_media_sha256'] and list(batch.image_grids[0])==plan['observed_image_grid_thw'],'actual prompt/media/grid identity')
                require(terminal['continuations']<28 and terminal['new_tokens']+3084<=86352,'shard generation bounds')
                tick=time.monotonic();result=generate_continuations(qwen.model,batch,extensions=[[]],budgets=[3084],eos_token_id=151645,pad_token_id=qwen.tokenizer.pad_token_id,trace='none')[0]
                require(result.request_id==frozen['example_id'],'native request association');terminal['continuations']+=1;terminal['new_tokens']+=len(result.token_ids)
                ids=list(result.token_ids);text=qwen.tokenizer.decode(ids,skip_special_tokens=False)
                row=dict(example_id=frozen['example_id'],split='dev112',manifest_sha256=digest(packet),action_ids=ids,prefix_ids=[],forced_ids=[],remaining_budget=3084,
                    text=text,stop_reason=result.stop_reason,seconds=time.monotonic()-tick,parsed=native_record(text,frozen['case'],frozen['baseline'],result.stop_reason))
                validate_natural(row,frozen,qwen.tokenizer);stream.write(json.dumps(row)+'\n');stream.flush();os.fsync(stream.fileno())
        cold=consume(path,packet,qwen.tokenizer,[r['example_id'] for r in records]);publish(run/'consumer.json',cold)
        require(terminal['continuations']==28,'complete shard28 required');terminal['status']='completed'
    except BaseException as exc:
        terminal.update(status='failed',error=repr(exc),traceback=traceback.format_exc());raise
    finally:
        signal.alarm(0);terminal.update(elapsed_seconds=time.monotonic()-started,rss_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0,
            artifact_bytes=sum(p.stat().st_size for p in run.rglob('*') if p.is_file()))
        publish(run/'terminal.json',terminal)


def require_complete_shards(terminals,packet):
    require(len(terminals)==4 and {t['shard'] for t in terminals}==set(range(4)),'unique four shard receipts required')
    for t in terminals:
        require(t['status']=='completed' and t['manifest_sha256']==digest(packet) and t['gpu']==GPUS[t['shard']] and
            t['continuations']==28 and t['model_loads']==1 and t['score_forwards']==0 and t['new_tokens']<=86352 and t['elapsed_seconds']<3600,'incomplete/wrong shard receipt')


def reduce_records(cold,packet):
    before=aggregate_scores([r['baseline_score'] for r in packet['records']]);after=aggregate_scores([r['score'] for r in cold])
    return dict(schema='selective_preservation.locked_dev112_reduction.v1',images=112,source=before,candidate=after,
        owner_changes={t:{k:sum(len(r['owner_changes'][t][k]) for r in cold) for k in ('gained','lost','retained')} for t in ('50','60','80')},
        eos=sum(r['stop_reason']=='im_end' for r in cold),per_image={r['example_id']:dict(owner_changes=r['owner_changes'],score=r['score'],matching_reassignment=r['matching_reassignment']) for r in cold},
        limitation='Locked candidate on112 historical development images excluding exposed guard16; not untouched confirmation. No new checkpoint selection.')


def merge(verify_only=False):
    from tokenizers import Tokenizer
    packet=json.loads((OUTPUT/'manifest.json').read_text());validate_packet(packet);locked_checkpoint()
    terminals=[json.loads((OUTPUT/f'shard-{i}/terminal.json').read_text()) for i in range(4)];require_complete_shards(terminals,packet)
    tokenizer=Tokenizer.from_file(packet['source_model']['base_model_path']+'/tokenizer.json');all_rows=[]
    for i in range(4):
        run=OUTPUT/f'shard-{i}';expected=[r['example_id'] for r in shard_records(packet,i)]
        cold=consume(run/'rows.jsonl',packet,tokenizer,expected)
        require(cold==json.loads((run/'consumer.json').read_text()),'fresh shard native consumer differs')
        require(sum(len(r['action_ids']) for r in cold)==terminals[i]['new_tokens'],'shard token count differs')
        all_rows.extend(cold)
    by_id=indexed(all_rows,'example_id');require(set(by_id)=={r['example_id'] for r in packet['records']} and len(by_id)==112,'merged exact112 IDs')
    ordered=[by_id[r['example_id']] for r in packet['records']];reduced=reduce_records(ordered,packet)
    if verify_only:
        require(ordered==json.loads((OUTPUT/'consumer.json').read_text()) and reduced==json.loads((OUTPUT/'reduction.json').read_text()),'merged cold replay differs')
    else:
        publish(OUTPUT/'consumer.json',ordered);publish(OUTPUT/'reduction.json',reduced)
        publish(OUTPUT/'resources.json',dict(model_loads=4,score_forwards=0,continuations=112,new_tokens=sum(t['new_tokens'] for t in terminals),
            allocated_gpu_seconds=sum(t['elapsed_seconds'] for t in terminals),shards=terminals))
    return dict(status='candidate_cpu_verified' if verify_only else 'candidate_merged',images=112,manifest_sha256=digest(packet),
        source50=reduced['source']['50'],candidate50=reduced['candidate']['50'],owner_changes=reduced['owner_changes'],
        consumer_sha256=file_hash(OUTPUT/'consumer.json'),reduction_sha256=file_hash(OUTPUT/'reduction.json'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','execute','merge','verify']);p.add_argument('--shard',type=int)
    args=p.parse_args()
    if args.command=='prepare':prepare()
    elif args.command=='execute':execute(args.shard)
    else:print(json.dumps(merge(verify_only=args.command=='verify'),indent=2))
