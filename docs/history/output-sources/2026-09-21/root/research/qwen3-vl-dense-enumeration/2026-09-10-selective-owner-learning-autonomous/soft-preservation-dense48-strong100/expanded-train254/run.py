"""Unchanged strong100: exact remaining acquisition-pool254 natural outputs."""
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
from probes.dora_owner_learning.entrance_ce_eval import consume,validate_natural,aggregate_scores
from probes.dora_owner_learning.selective_preservation_strong_eval import ARM_ROOT,validate_receipt
from probes.dora_owner_learning.route_access import CONFIG,checkpoint_config,checked_ids,publish
from probes.dora_owner_learning.round1_realization import SOURCE_ROOT
from probes.dora_owner_learning.reward_rows import _gt_objects
from probes.source_rweak_row_cross.owner_row_robustness import native_record
from probes.source_rweak_row_cross.run import build_requests

OUTPUT=Path(__file__).resolve().parent
GPUS=tuple(range(8))
COUNTS=(32,32,32,32,32,32,31,31)
RECEIPT_SHA='037f1af3561e5522837364c41266af135e20c3d80579767092c63b761d6821ba'
TARGETS={368,7116}


def partition(train_ids,support_ids,dev_ids):
    train=list(map(int,train_ids));support=list(map(int,support_ids));dev=list(map(int,dev_ids))
    require(len(train)==len(set(train))==256 and TARGETS<=set(train),'exact Source train256 IDs')
    require(len(dev)==len(set(dev))==128 and not set(train)&set(dev),'train/dev identity overlap')
    require(len(support)==len(set(support))==48 and set(support)<=set(train)-TARGETS,'support48 identity/overlap')
    selected=sorted(set(train)-TARGETS);outside=sorted(set(selected)-set(support));shards=[selected[i::8] for i in range(8)]
    require(len(selected)==254 and len(outside)==206 and [len(s) for s in shards]==list(COUNTS),'254/48/206/shard counts')
    return selected,outside,shards


def locked_checkpoint():
    path=ARM_ROOT/'training/receipt.json';require(file_hash(path)==RECEIPT_SHA,'locked strong100 receipt changed')
    receipt=json.loads(path.read_text());old=json.loads((ARM_ROOT/'evaluation/manifest.json').read_text())
    return receipt,validate_receipt(receipt,old),old


def prepare():
    import yaml
    from src.data.geometry import parse_coord_token
    from src.qwen.runtime_loading import QwenLoadOptions,load_qwen_components_from_options
    from src.qwen.native import prepare_native_inputs
    require(not (OUTPUT/'manifest.json').exists() and not (OUTPUT/'selection.json').exists(),'occupied preparation')
    receipt,adapter,old=locked_checkpoint();config=yaml.safe_load((SOURCE_ROOT/'configs/resolved.yaml').read_text())['config']
    inputs=rows(config['data']['input_jsonl']);raw=indexed(rows(SOURCE_ROOT/'gt_vs_pred.jsonl'),'row_id')
    images=indexed(rows(SOURCE_ROOT/'image_plan.jsonl'),'row_id');manifest=json.loads((SOURCE_ROOT/'run_manifest.json').read_text());prompts=indexed(manifest['prompt_trace'],'row_id')
    support=list(map(int,receipt['support_image_ids']));dev=[r['image_id'] for r in old['records'] if r['split']!='train']
    selected,outside,shards=partition([r['image_id'] for r in inputs],support,dev)
    selection=dict(image_ids=selected,support48=sorted(support),outside206=outside,already_read_train2=sorted(TARGETS),dev128=dev,
        shards=[dict(shard=i,gpu=GPUS[i],image_ids=s) for i,s in enumerate(shards)],
        rule='All original Source train256 minus368/7116, no EOS/drop/category/density/outcome filtering; numeric-ID roundrobin.')
    publish(OUTPUT/'selection.json',selection)
    policy=manifest['generation_policy'];require(not policy['do_sample'] and policy['max_new_tokens']==3084 and policy['repetition_penalty']==1 and policy['top_p']==1,'Source policy')
    require(config['model']['dtype']=='fp32' and config['backend']['hf']==dict(attn_implementation='sdpa',patch_embed_linearization='enabled'),'Source numerics')
    require(config['model']['base_model']==old['source_model']['base_model_path'] and config['adapter']['path']==old['source_model']['current_adapter']['root'] and config['embedding_delta']['path']==old['source_model']['source_embedding']['root'],'Source model/config identity')
    traces=defaultdict(list)
    for t in rows(SOURCE_ROOT/'pred_token_trace.jsonl'):
        if t['trace_type']=='generated_token' and not t['is_pad']:traces[t['row_id']].append(t)
    qwen=load_qwen_components_from_options(QwenLoadOptions(base_model=config['model']['base_model'],dtype='fp32',attn_implementation='sdpa',patch_embed_linearization='enabled',load_model=False))
    by_id={int(r['image_id']):r for r in inputs};records=[]
    for iid in selected:
        inp=by_id[iid];eid=f'coco2017_train_{iid:012d}';golden=raw[eid];plan=images[eid]
        gt=[dict(description=o['desc'],bbox=[parse_coord_token(v,field='GT') for v in o['bbox_2d']],object_id=str(o['coco_ann_id'])) for o in inp['objects']]
        require(_gt_objects(dict(golden,gt=gt),row_id=eid)==_gt_objects(golden,row_id=eid) and [g['object_id'] for g in gt]==[str(g['object_id']) for g in golden['gt']],'Source/input GT identity')
        require(file_hash(golden['image_path'])==plan['image_content_sha256'],'Source image bytes')
        case=dict(row_id=eid,row_index=golden['row_index'],image_path=golden['image_path'],image_width=golden['image_width'],image_height=golden['image_height'],input_record=inp,image_plan=plan)
        requests,_=build_requests(qwen,config,[case]);batch=prepare_native_inputs(qwen.processor,requests,device='cpu',record_media_identity=True)
        prompt=list(batch.prompt_token_ids[0]);require(digest(prompt)==prompts[eid]['backend_executed_prompt_token_ids_sha256'],'Source prompt tokens')
        require(batch.media_sha256[0]==plan['executed_media_sha256'] and list(batch.image_grids[0])==plan['observed_image_grid_thw'],'Source media/grid identity')
        trace=sorted(traces[eid],key=lambda t:t['generated_step_index']);require([t['generated_step_index'] for t in trace]==list(range(len(trace))),'Source trace coverage')
        ids=checked_ids([t['token_id'] for t in trace],golden['decode_stop_reason'])
        require(qwen.tokenizer.decode(ids,skip_special_tokens=False)==golden['raw_decode_text'] and native_record(golden['raw_decode_text'],case,golden,golden['decode_stop_reason'])==golden,'Source token/parser identity')
        records.append(dict(example_id=eid,image_id=iid,split='support48' if iid in support else 'outside206',case=case,prompt_token_ids=prompt,baseline=golden,baseline_ids=ids,
            baseline_score=score(golden,seed=-1,length=len(ids),stop=golden['decode_stop_reason'])))
        del batch
    prior_path=ARM_ROOT/'evaluation/consumer.json';prior=json.loads(prior_path.read_text());require([r['example_id'] for r in prior]==[r['example_id'] for r in old['records']],'already-read130 identity')
    for r,frozen in zip(prior,old['records'],strict=True):
        validate_natural(r,frozen,qwen.tokenizer);require(score(r['parsed'],seed=-1,length=len(r['action_ids']),stop=r['stop_reason'])==r['score'],'already-read candidate native score identity')
    paths=[SOURCE_ROOT/'run_manifest.json',SOURCE_ROOT/'configs/resolved.yaml',SOURCE_ROOT/'gt_vs_pred.jsonl',SOURCE_ROOT/'image_plan.jsonl',SOURCE_ROOT/'pred_token_trace.jsonl',Path(config['data']['input_jsonl']),
        ARM_ROOT/'evaluation/manifest.json',prior_path,ARM_ROOT/'training/receipt.json']
    packet=dict(schema='strong100.expanded_train254.v1',selection=selection,records=records,config=config,source_model=old['source_model'],prior_records=old['records'],prior_consumer=prior,
        training_receipt_sha256=RECEIPT_SHA,source_files={str(p):file_hash(p) for p in paths},producer_sha256=file_hash(__file__))
    publish(OUTPUT/'manifest.json',packet)
    print(json.dumps(dict(status='frozen',records=254,outside206=206,support48=48,shards=[len(s) for s in shards],manifest_sha256=digest(packet),
        Source_caps=sum(r['baseline_score']['cap'] for r in records),Source_parser_drops=sum(r['baseline_score']['parser_drops'] for r in records))))


def validate_packet(packet):
    require(packet['schema']=='strong100.expanded_train254.v1' and packet['training_receipt_sha256']==RECEIPT_SHA,'wrong locked expansion packet')
    s=packet['selection'];selected,outside,shards=partition(s['image_ids']+s['already_read_train2'],s['support48'],s['dev128'])
    require(s['already_read_train2']==sorted(TARGETS) and s['image_ids']==selected and s['outside206']==outside,'changed strata')
    require(s['shards']==[dict(shard=i,gpu=GPUS[i],image_ids=x) for i,x in enumerate(shards)],'changed shard assignment')
    require([r['image_id'] for r in packet['records']]==selected and len(indexed(packet['records'],'example_id'))==254,'exact254 records')
    for r in packet['records']:require(r['split']==('support48' if r['image_id'] in s['support48'] else 'outside206'),'stratum identity mismatch')
    require(packet['prior_records']==json.loads((ARM_ROOT/'evaluation/manifest.json').read_text())['records'] and packet['prior_consumer']==json.loads((ARM_ROOT/'evaluation/consumer.json').read_text()),'prior130 changed')
    for p,sha in packet['source_files'].items():require(file_hash(p)==sha,'Source/control/receipt bytes changed')


def shard_records(packet,shard):
    require(type(shard) is int and 0<=shard<8,'invalid shard')
    ids=set(packet['selection']['shards'][shard]['image_ids']);r=[r for r in packet['records'] if r['image_id'] in ids]
    require(len(r)==COUNTS[shard] and len({x['example_id'] for x in r})==COUNTS[shard],'exact assigned shard IDs')
    return r


def require_complete_shards(terminals,packet):
    require(len(terminals)==8 and {t['shard'] for t in terminals}==set(range(8)),'eightunique complete receipts required')
    for t in terminals:
        i=t['shard'];require(t['status']=='completed' and t['manifest_sha256']==digest(packet) and t['gpu']==i and t['continuations']==COUNTS[i] and t['model_loads']==1 and t['score_forwards']==0
            and t['new_tokens']<=COUNTS[i]*3084 and t['elapsed_seconds']<3600 and t['training_receipt_sha256']==RECEIPT_SHA,'wrong/incomplete shard receipt')


def reduce_records(cold,packet):
    new=indexed(cold,'example_id');prior=indexed(packet['prior_consumer'],'example_id');require(not set(new)&set(prior),'new/prior overlap')
    all_rows={**new,**prior};sources=indexed(packet['records']+packet['prior_records'],'example_id');require(len(all_rows)==len(sources)==384,'descriptive384 union coverage')
    ids=[r['example_id'] for r in packet['records']];old_train=[r['example_id'] for r in packet['prior_records'] if r['split']=='train'];old_dev=[r['example_id'] for r in packet['prior_records'] if r['split']!='train']
    panels=dict(outside206=[r['example_id'] for r in packet['records'] if r['split']=='outside206'],support48=[r['example_id'] for r in packet['records'] if r['split']=='support48'],
        remaining254=ids,train256_descriptive=ids+old_train,already_read_train2=old_train,already_read_dev128=old_dev,union384_descriptive=ids+old_train+old_dev)
    result={}
    for label,panel in panels.items():
        selected=[all_rows[eid] for eid in panel];result[label]=dict(images=len(panel),source=aggregate_scores([sources[eid]['baseline_score'] for eid in panel]),
            strong100=aggregate_scores([r['score'] for r in selected]),owner_changes={t:{k:sum(len(r['owner_changes'][t][k]) for r in selected) for k in ('gained','lost','retained')} for t in ('50','60','80')},
            eos=sum(r['stop_reason']=='im_end' for r in selected))
    return dict(schema='strong100.expanded_train254_reduction.v1',new_images=254,panels=result,
        per_image={eid:dict(stratum=sources[eid]['split'],source=sources[eid]['baseline_score'],strong100=r['score'],owner_changes=r['owner_changes'],matching_reassignment=r['matching_reassignment']) for eid,r in new.items()},
        limitation='Existing acquisition pool; outside206 excludes this update support but is not untouched heldout. Priortrain2/dev128 reused descriptively, no new positive labels or confirmation data.')


def merge(verify_only=False):
    from tokenizers import Tokenizer
    packet=json.loads((OUTPUT/'manifest.json').read_text());validate_packet(packet);locked_checkpoint()
    terminals=[json.loads((OUTPUT/f'shard-{i}/terminal.json').read_text()) for i in range(8)];require_complete_shards(terminals,packet)
    exits=json.loads((OUTPUT/'process-exits.json').read_text())['results'];require(len(exits)==8 and {r['shard'] for r in exits}==set(range(8)) and all(r['exit_code']==0 for r in exits),'eight exit0 processes required')
    tokenizer=Tokenizer.from_file(packet['source_model']['base_model_path']+'/tokenizer.json');collected=[]
    for i in range(8):
        run=OUTPUT/f'shard-{i}';cold=consume(run/'rows.jsonl',packet,tokenizer,[r['example_id'] for r in shard_records(packet,i)])
        require(cold==json.loads((run/'consumer.json').read_text()) and sum(len(r['action_ids']) for r in cold)==terminals[i]['new_tokens'],'shard native consumer/counter mismatch');collected.extend(cold)
    by_id=indexed(collected,'example_id');require(set(by_id)=={r['example_id'] for r in packet['records']} and len(by_id)==254,'exact merged254 IDs')
    ordered=[by_id[r['example_id']] for r in packet['records']];reduced=reduce_records(ordered,packet)
    if verify_only:
        require(ordered==json.loads((OUTPUT/'consumer.json').read_text()) and reduced==json.loads((OUTPUT/'reduction.json').read_text()),'merged native/reduction replay mismatch')
    else:
        publish(OUTPUT/'consumer.json',ordered);publish(OUTPUT/'reduction.json',reduced)
        publish(OUTPUT/'resources.json',dict(model_loads=8,score_forwards=0,continuations=254,new_tokens=sum(t['new_tokens'] for t in terminals),allocated_gpu_seconds=sum(t['elapsed_seconds'] for t in terminals),shards=terminals))
    return dict(status='candidate_cpu_verified' if verify_only else 'candidate_merged',images=254,manifest_sha256=digest(packet),
        consumer_sha256=file_hash(OUTPUT/'consumer.json'),reduction_sha256=file_hash(OUTPUT/'reduction.json'))
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
        model_forwards=0,image_forwards=0,continuations=0,new_tokens=0,producer_sha256=file_hash(__file__),training_receipt_sha256=RECEIPT_SHA)
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
                require(terminal['continuations']<len(records) and terminal['new_tokens']+3084<=3084*len(records),'shard generation bounds')
                tick=time.monotonic();result=generate_continuations(qwen.model,batch,extensions=[[]],budgets=[3084],eos_token_id=151645,pad_token_id=qwen.tokenizer.pad_token_id,trace='none')[0]
                require(result.request_id==frozen['example_id'],'native request association');terminal['continuations']+=1;terminal['new_tokens']+=len(result.token_ids)
                ids=list(result.token_ids);text=qwen.tokenizer.decode(ids,skip_special_tokens=False)
                row=dict(example_id=frozen['example_id'],split=frozen['split'],manifest_sha256=digest(packet),action_ids=ids,prefix_ids=[],forced_ids=[],remaining_budget=3084,
                    text=text,stop_reason=result.stop_reason,seconds=time.monotonic()-tick,parsed=native_record(text,frozen['case'],frozen['baseline'],result.stop_reason))
                validate_natural(row,frozen,qwen.tokenizer);stream.write(json.dumps(row)+'\n');stream.flush();os.fsync(stream.fileno())
        cold=consume(path,packet,qwen.tokenizer,[r['example_id'] for r in records]);publish(run/'consumer.json',cold)
        require(terminal['continuations']==len(records),'complete assigned shard required');terminal['status']='completed'
    except BaseException as exc:
        terminal.update(status='failed',error=repr(exc),traceback=traceback.format_exc());raise
    finally:
        signal.alarm(0);terminal.update(elapsed_seconds=time.monotonic()-started,rss_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0,
            artifact_bytes=sum(p.stat().st_size for p in run.rglob('*') if p.is_file()))
        publish(run/'terminal.json',terminal)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','execute','merge','verify']);p.add_argument('--shard',type=int);args=p.parse_args()
    if args.command=='prepare':prepare()
    elif args.command=='execute':execute(args.shard)
    else:print(json.dumps(merge(verify_only=args.command=='verify'),indent=2))
