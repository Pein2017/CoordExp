"""Frozen support100 natural130 read against Source, KL10, wide31 and dense48."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import resource
import signal
import time
import traceback

from .candidate_opportunity import digest,file_hash,indexed,require,rows
from .entrance_ce_eval import validate_receipt as validate_source_receipt,consume,validate_natural,aggregate_scores,owner_change
from .selective_preservation_dense_eval import (
    validate_global_completion as validate_rank_completion,validate_numerical_admission as validate_prior_path,
    reduce_records as prior_reduction,
)
from .selective_preservation_eval import target_boxes
from .branch_bridge import summarize_logits
from .route_access import CONFIG,checkpoint_config,publish
from .round1_realization import ROOT
from probes.source_rweak_row_cross.owner_row_robustness import native_record
from probes.source_rweak_row_cross.run import build_requests

ARM_ROOT=ROOT/'2026-09-10-selective-owner-learning-autonomous/soft-preservation-dense48-strong100'
TRAINING=ARM_ROOT/'training'
OUTPUT=ARM_ROOT/'evaluation'
DENSE_ROOT=ARM_ROOT.parent/'soft-preservation-dense48'
PRIOR_RECEIPT=DENSE_ROOT/'training-retry1/receipt.json'
PRIOR_RECEIPT_SHA='5b14c1da1f64cb9413df0b36909ce9e3711fd90cbae7cdbb89f135fc7c8663e2'
ADMISSION='ddp_path_reuse_support100_v1'
GPUS=(1,3,4,5)
COUNTS=(33,33,32,32)


def prepare():
    require(not OUTPUT.exists(),'occupied strong evaluation output')
    path=DENSE_ROOT/'evaluation-retry1/manifest.json';old=json.loads(path.read_text());controls=DENSE_ROOT/'evaluation-retry1/consumer.json'
    packet=dict(old);packet.update(schema='selective_preservation_strong.eval.v1',arm='soft-preservation-dense48-strong100',
        attempt='support100',numerical_admission=ADMISSION,dense48_controls=json.loads(controls.read_text()),
        frozen_dense_manifest_sha256=file_hash(path),support_coefficient=100.)
    packet['source_files']=dict(old['source_files'],**{str(path):file_hash(path),str(controls):file_hash(controls)})
    validate_packet(packet);OUTPUT.mkdir(parents=True);publish(OUTPUT/'manifest.json',packet)
    print(json.dumps(dict(status='frozen',manifest_sha256=digest(packet),images=130,shard_counts=COUNTS,GPUs=GPUS)))


def validate_packet(packet):
    require(packet['schema']=='selective_preservation_strong.eval.v1' and packet['arm']=='soft-preservation-dense48-strong100' and
        packet['numerical_admission']==ADMISSION and packet['support_coefficient']==100.,'wrong support100 packet')
    path=DENSE_ROOT/'evaluation-retry1/manifest.json';require(file_hash(path)==packet['frozen_dense_manifest_sha256'],'original130 packet changed')
    old=json.loads(path.read_text())
    for k in ('records','shards','configs','source_model','kl10_controls','wide31_controls','support_image_ids','support_action_states'):
        require(packet[k]==old[k],'changed frozen130/support payload')
    require(packet['dense48_controls']==json.loads((DENSE_ROOT/'evaluation-retry1/consumer.json').read_text()),'dense48 control changed')
    r=packet['records'];require(len(r)==130 and len(indexed(r,'example_id'))==130 and [x['split'] for x in r]==['train']*2+['guard']*16+['dev112']*112,'exact130split identity')
    require(packet['shards']==[dict(shard=i,gpu=GPUS[i],example_ids=[x['example_id'] for x in r[i::4]]) for i in range(4)],'roundrobin shards changed')
    for p,sha in packet['source_files'].items():require(file_hash(p)==sha,'frozen Source/control changed')


def validate_admission(receipt):
    require(receipt['numerical_admission']==ADMISSION,'wrong DDP-path admission')
    prior=receipt['prior_path_proof'];require(Path(prior['path']).resolve()==PRIOR_RECEIPT.resolve() and prior['sha256']==PRIOR_RECEIPT_SHA and file_hash(PRIOR_RECEIPT)==PRIOR_RECEIPT_SHA,'wrong accepted prior path proof')
    # The old numerical equality applies ONLY to the accepted prior coefficient10 path.
    validate_prior_path(json.loads(PRIOR_RECEIPT.read_text()))
    fixture=receipt['coefficient_fixture'];path=Path(fixture['path']);require(path.resolve().parent==TRAINING.resolve() and file_hash(path)==fixture['sha256'],'coefficient fixture path/bytes')
    proof=json.loads(path.read_text());require(proof['status']=='passed' and proof['support_gradient_ratio']==10 and proof['old_two_unchanged'] is True and proof['compensation_and_clip_checks'] is True,'coefficient-gradient fixture failed')
    smoke=receipt['two_step'];path=TRAINING/'two-step-smoke.json';require(Path(smoke['path']).resolve()==path.resolve() and file_hash(path)==smoke['sha256'],'current two-step proof path/bytes')
    s=json.loads(path.read_text());require(s['status']=='passed' and s['updates']==[1,2] and s['reduced_gradient_adapter_optimizer_rank_identity'] is True and s['frozen_bytes_unchanged'] is True,'current two-step identity failed')
    require(s['initial_KL_zero'] is True and s['initial_reference_count']==50,'current initial reference KL/count failed')
    require(set(s['step2_max_KL_by_rank'])==set(map(str,range(8))) and any(v>1e-8 for v in s['step2_max_KL_by_rank'].values()),'current step2 nonzero KL missing')


def validate_receipt(receipt,packet):
    require(receipt['schema_version']=='selective_preservation_strong.training.v1' and receipt['status']=='completed','wrong/incomplete strong receipt')
    require(receipt['updates']==23 and receipt['stop_reason']=='fixed_steps' and receipt['lambda_kl']==10. and receipt['lambda_support_kl']==100.,'wrong fixed23 support100 recipe')
    require(receipt['support_image_ids']==packet['support_image_ids'] and len(set(receipt['support_image_ids']))==48 and receipt['support_action_states']==packet['support_action_states']==5848,'wrong support48/fullstates')
    require(Path(receipt['adapter']['root']).resolve()==(TRAINING/'adapter').resolve() and file_hash(TRAINING/'inputs.json')==receipt['inputs_sha256'],'wrong saved candidate/inputs')
    validate_rank_completion(receipt);uses=[]
    for r in receipt['rank_terminals']:uses.append(json.loads(Path(r['path']).read_text())['counters']['support_KL_trajectories'])
    require(uses==[138]*8 and sum(uses)==receipt['actual_support_uses']==1104,'actual support-use accounting')
    validate_admission(receipt)
    return validate_source_receipt(receipt,packet)


def locked_checkpoint():
    packet=json.loads((OUTPUT/'manifest.json').read_text());path=TRAINING/'receipt.json';receipt=json.loads(path.read_text())
    return receipt,validate_receipt(receipt,packet),file_hash(path)


def shard_records(packet,shard):
    require(type(shard) is int and 0<=shard<4,'invalid shard')
    by_id=indexed(packet['records'],'example_id');r=[by_id[eid] for eid in packet['shards'][shard]['example_ids']]
    require(len(r)==COUNTS[shard] and len({x['example_id'] for x in r})==COUNTS[shard],'exact shard coverage')
    return r


def reduce_records(cold,packet):
    result=prior_reduction(cold,packet);result['schema']='selective_preservation_strong.eval_reduction.v1'
    old=indexed(packet['records'],'example_id');dense=indexed(packet['dense48_controls'],'example_id')
    for split,s in result['splits'].items():
        selected=[r for r in cold if old[r['example_id']]['split']==split or split=='dev128_descriptive' and old[r['example_id']]['split']!='train']
        s['strong100']=s.pop('dense48');s['dense48']=aggregate_scores([dense[r['example_id']]['score'] for r in selected])
        s['versus']['dense48']={t:{k:sum(len(owner_change(r['score'][t]['owners'],dense[r['example_id']]['score'][t]['owners'])[k]) for r in selected) for k in ('gained','lost','retained')} for t in ('50','60','80')}
    for eid,p in result['per_image'].items():
        p['strong100']=p.pop('dense48');p['dense48']=dense[eid]['score']
        p['versus_dense48']={t:owner_change(p['strong100'][t]['owners'],dense[eid]['score'][t]['owners']) for t in ('50','60','80')}
        if p['split']=='train':
            p['target_boxes']['strong100']=p['target_boxes'].pop('dense48');owner=old[eid]['bridge_case']['owner']
            p['target_boxes']['dense48']=target_boxes(dense[eid]['parsed'],dense[eid]['score'],owner)
    return result
def score_terminal_entries(qwen,packet,receipt,run):
    import numpy as np
    import torch
    from src.qwen.native import prepare_native_inputs,prepare_replay
    for frozen in packet['records'][:2]:
        request,_=build_requests(qwen,packet['configs']['train'],[frozen['case']]);batch=prepare_native_inputs(qwen.processor,request,device=torch.device('cuda:0'),record_media_identity=True)
        require(list(batch.prompt_token_ids[0])==frozen['prompt_token_ids'] and batch.media_sha256[0]==frozen['case']['image_plan']['executed_media_sha256'],'cold entry media/prompt')
        ent=frozen['entrance'];cid=frozen['bridge_case']['case_id'];stem=cid.replace(':','_')
        with torch.inference_mode():
            replay=prepare_replay(qwen.model,batch.inputs,prompt_token_ids=frozen['prompt_token_ids'],continuation_token_ids=ent['state_ids']+[ent['A_id']])
            logits=replay.aligned_logits(qwen.model(**replay.inputs).logits)
            require(logits.shape[0]==ent['action_index']+1 and int(replay.target_ids[-1])==ent['A_id'],'cold causal entry')
            values=logits[-1].detach().float().cpu().numpy().copy();del logits,replay
        path=run/f'{stem}-cold-entry-logits.npy'
        with path.open('xb') as stream:np.save(stream,values,allow_pickle=False)
        observed=summarize_logits(np.load(path,allow_pickle=False),ent);saved=receipt['final_scores'][cid]
        deltas={k:observed[k]-saved[k] for k in ('logprob','A_vs_best_other_margin')}
        require(all(abs(v)<=1e-5 for v in deltas.values()) and observed['A_id']==saved['target_id'] and all(observed[k]==saved[k] for k in ('rank_min','top1_id')),'strong saved/cold entry mismatch')
        publish(run/f'{stem}-cold-entry.json',dict(readout=observed,trainer=saved,deltas=deltas,tolerance=1e-5,logits_sha256=file_hash(path)))
    return 2

def require_complete_shards(terminals,packet):
    require(len(terminals)==4 and {t['shard'] for t in terminals}==set(range(4)),'fourunique receipts required')
    for t in terminals:
        i=t['shard'];require(t['status']=='completed' and t['manifest_sha256']==digest(packet) and t['gpu']==GPUS[i] and t['continuations']==COUNTS[i]
            and t['model_loads']==1 and t['score_forwards']==(2 if i==0 else 0) and t['new_tokens']<=COUNTS[i]*3084 and t['elapsed_seconds']<3600,'incomplete/wrong strong shard')
    require(len({t['training_receipt_sha256'] for t in terminals})==1,'mixed candidate receipts')

def render_train2(cold):
    from src.vis import render_gt_vs_prediction
    projection=OUTPUT/'visualization-input';projection.mkdir(exist_ok=False);train=[r for r in cold if r['split']=='train']
    for name in ('gt_vs_pred.jsonl','gt_vs_pred_scored.jsonl'):
        with (projection/name).open('x') as stream:
            for r in train:stream.write(json.dumps(r['parsed'])+'\n')
    publish(projection/'provenance.json',dict(source=str(OUTPUT/'consumer.json'),scope='Unchanged native geometry projection, no invented likelihood scores.'))
    require(not (OUTPUT/'visualizations').exists(),'occupied visual output')
    result=render_gt_vs_prediction(projection,OUTPUT/'visualizations',duplicate_iou_threshold=.95)
    require(len(result.image_paths)==2,'train2 visualization coverage')

def merge(verify_only=False):
    import numpy as np
    from tokenizers import Tokenizer
    packet=json.loads((OUTPUT/'manifest.json').read_text());validate_packet(packet);training,_,sha=locked_checkpoint()
    terminals=[json.loads((OUTPUT/f'shard-{i}/terminal.json').read_text()) for i in range(4)];require_complete_shards(terminals,packet)
    exits=json.loads((OUTPUT/'process-exits.json').read_text())['results'];require(len(exits)==4 and {r['shard'] for r in exits}==set(range(4)) and all(r['exit_code']==0 for r in exits),'fourcomplete exit0 required')
    require(all(t['training_receipt_sha256']==sha for t in terminals),'sealed receipt changed')
    tokenizer=Tokenizer.from_file(packet['source_model']['base_model_path']+'/tokenizer.json');all_rows=[]
    for i in range(4):
        run=OUTPUT/f'shard-{i}';cold=consume(run/'rows.jsonl',packet,tokenizer,[r['example_id'] for r in shard_records(packet,i)])
        require(cold==json.loads((run/'consumer.json').read_text()) and sum(len(r['action_ids']) for r in cold)==terminals[i]['new_tokens'],'shard native consumer/counter mismatch')
        all_rows.extend(cold)
    by_id=indexed(all_rows,'example_id');require(set(by_id)=={r['example_id'] for r in packet['records']} and len(by_id)==130,'exact merged130IDs')
    ordered=[by_id[r['example_id']] for r in packet['records']];reduced=reduce_records(ordered,packet)
    for r in packet['records'][:2]:
        cid=r['bridge_case']['case_id'];stem=cid.replace(':','_');run=OUTPUT/'shard-0';saved=json.loads((run/f'{stem}-cold-entry.json').read_text());path=run/f'{stem}-cold-entry-logits.npy'
        require(file_hash(path)==saved['logits_sha256'] and summarize_logits(np.load(path,allow_pickle=False),r['entrance'])==saved['readout'] and saved['trainer']==training['final_scores'][cid],'cold entry arithmetic/identity mismatch')
    if verify_only:
        require(ordered==json.loads((OUTPUT/'consumer.json').read_text()) and reduced==json.loads((OUTPUT/'reduction.json').read_text()),'merged CPU consumer/reduction mismatch')
        require(len(list((OUTPUT/'visualizations').glob('*.png')))==2,'train2 visual coverage')
    else:
        publish(OUTPUT/'consumer.json',ordered);publish(OUTPUT/'reduction.json',reduced);render_train2(ordered)
        publish(OUTPUT/'resources.json',dict(model_loads=4,score_forwards=2,continuations=130,new_tokens=sum(t['new_tokens'] for t in terminals),
            allocated_gpu_seconds=sum(t['elapsed_seconds'] for t in terminals),shards=terminals))
    return dict(status='candidate_cpu_verified' if verify_only else 'candidate_merged',images=130,manifest_sha256=digest(packet),
        consumer_sha256=file_hash(OUTPUT/'consumer.json'),reduction_sha256=file_hash(OUTPUT/'reduction.json'))

def execute(shard):
    import torch
    from src.config.inference import load_research_infer_config
    from src.qwen.native import prepare_native_inputs
    from src.qwen.generation import generate_continuations
    from probes.dora_owner_learning.runtime import load_policy
    packet=json.loads((OUTPUT/'manifest.json').read_text());validate_packet(packet);records=shard_records(packet,shard)
    receipt,adapter,receipt_sha=locked_checkpoint();require(os.environ.get('CUDA_VISIBLE_DEVICES')==str(GPUS[shard]) and torch.cuda.device_count()==1,'assigned single GPU only')
    run=OUTPUT/f'shard-{shard}';run.mkdir(exist_ok=False);started=time.monotonic()
    terminal=dict(status='running',pid=os.getpid(),shard=shard,gpu=GPUS[shard],manifest_sha256=digest(packet),model_loads=0,score_forwards=0,
        model_forwards=0,image_forwards=0,continuations=0,new_tokens=0,producer_sha256=file_hash(__file__),training_receipt_sha256=receipt_sha)
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
        if shard==0:terminal['score_forwards']=score_terminal_entries(qwen,packet,receipt,run)
        path=run/'rows.jsonl'
        with path.open('x') as stream:
            for frozen in records:
                request,_=build_requests(qwen,packet['configs'][frozen['split']],[frozen['case']])
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
        require(terminal['continuations']==len(records),'complete assigned strong shard required');terminal['status']='completed'
    except BaseException as exc:
        terminal.update(status='failed',error=repr(exc),traceback=traceback.format_exc());raise
    finally:
        signal.alarm(0);terminal.update(elapsed_seconds=time.monotonic()-started,rss_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0,
            artifact_bytes=sum(p.stat().st_size for p in run.rglob('*') if p.is_file()))
        publish(run/'terminal.json',terminal)

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','execute','merge','verify']);p.add_argument('--shard',type=int);args=p.parse_args()
    if args.command=='prepare':prepare()
    elif args.command=='execute':execute(args.shard)
    else:print(json.dumps(merge(verify_only=args.command=='verify'),indent=2))


if __name__=='__main__':main()
