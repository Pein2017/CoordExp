"""Frozen Source/KL10/wide31 natural130 evaluation with isolated native shards."""
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
from .selective_preservation_eval import target_boxes,ARM_ROOT as KL10_ROOT
from .branch_bridge import summarize_logits
from .route_access import CONFIG,checkpoint_config,publish
from .round1_realization import ROOT
from probes.source_rweak_row_cross.owner_row_robustness import native_record
from probes.source_rweak_row_cross.run import build_requests

ARM_ROOT=ROOT/'2026-09-10-selective-owner-learning-autonomous/soft-preservation-wide31'
OUTPUT=ARM_ROOT/'evaluation'
PREFLIGHT=ARM_ROOT.parent/'support32-preflight.json'
GPUS=(1,3,4,5)
COUNTS=(33,33,32,32)


def source_packets():
    return json.loads((KL10_ROOT/'evaluation/manifest.json').read_text()),json.loads((KL10_ROOT/'dev112/manifest.json').read_text())


def expected_shards(records):
    ids=[r['example_id'] for r in records];require(len(ids)==len(set(ids))==130,'exact unique130 required')
    require([r['split'] for r in records]==['train']*2+['guard']*16+['dev112']*112,'frozen split/order changed')
    return [dict(shard=i,gpu=GPUS[i],example_ids=ids[i::4]) for i in range(4)]


def prepare():
    require(not OUTPUT.exists(),'occupied wide evaluation output')
    old18,old112=source_packets();records=old18['records']+old112['records'];shards=expected_shards(records)
    controls=json.loads((KL10_ROOT/'evaluation/execution/consumer.json').read_text())+json.loads((KL10_ROOT/'dev112/consumer.json').read_text())
    by_id=indexed(controls,'example_id');require(set(by_id)=={r['example_id'] for r in records},'KL10 control130 identity')
    require(len({r['image_id'] for r in records})==130,'image overlap')
    for r in records:
        c=by_id[r['example_id']];require(c['split']==r['split'] and c['parsed']['gt']==r['baseline']['gt'],'control split/GT identity')
    preflight=json.loads(PREFLIGHT.read_text());support=preflight['proposed_support31']['retained_image_ids']
    require(len(set(support))==31 and not set(map(int,support))&{r['image_id'] for r in records},'support/evaluation separation')
    paths=[KL10_ROOT/'evaluation/manifest.json',KL10_ROOT/'dev112/manifest.json',KL10_ROOT/'evaluation/execution/consumer.json',KL10_ROOT/'dev112/consumer.json',PREFLIGHT]
    packet=dict(schema='selective_preservation_wide.eval.v1',arm='soft-preservation-wide31',records=records,shards=shards,
        configs=dict(old18['configs'],dev112=old112['config']),source_model=old18['source_model'],kl10_controls=[by_id[r['example_id']] for r in records],
        support_image_ids=support,support_preflight_sha256=file_hash(PREFLIGHT),source_files={**old18['source_files'],**old112['source_files'],**{str(p):file_hash(p) for p in paths}},
        scope='Same130 exposeddevelopment images; fourshards of onecandidate, no newselection or confirmation.')
    validate_packet(packet);OUTPUT.mkdir(parents=True);publish(OUTPUT/'manifest.json',packet)
    print(json.dumps(dict(status='frozen',manifest_sha256=digest(packet),images=130,shard_counts=[len(s['example_ids']) for s in shards],GPUs=GPUS)))


def validate_packet(packet):
    require(packet['schema']=='selective_preservation_wide.eval.v1' and packet['arm']=='soft-preservation-wide31','wrong wide arm/schema')
    old18,old112=source_packets();require(packet['records']==old18['records']+old112['records'],'Source130 records changed')
    require(packet['source_model']==old18['source_model'] and packet['configs']==dict(old18['configs'],dev112=old112['config']),'Source model/config identity changed')
    require(packet['shards']==expected_shards(packet['records']),'changed wide shard assignment')
    controls=indexed(packet['kl10_controls'],'example_id');require(set(controls)=={r['example_id'] for r in packet['records']},'KL10 comparison coverage')
    for r in packet['records']:require(controls[r['example_id']]['split']==r['split'] and controls[r['example_id']]['parsed']['gt']==r['baseline']['gt'],'KL10 split/GT identity')
    retained=json.loads((KL10_ROOT/'evaluation/execution/consumer.json').read_text())+json.loads((KL10_ROOT/'dev112/consumer.json').read_text())
    require(packet['kl10_controls']==retained,'retained KL10 outcomes changed')
    for p,sha in packet['source_files'].items():require(file_hash(p)==sha,'frozen Source/control file changed')


def validate_receipt(receipt,packet):
    require(receipt['schema_version']=='selective_preservation_wide.training.v1' and receipt['status']=='completed','wrong/incomplete wide receipt')
    require(receipt['updates']==23 and receipt['stop_reason']=='fixed_steps','wide fixed23 required')
    require(receipt['lambda_kl']==receipt['lambda_support_kl']==10.,'both frozen KL weights10 required')
    require(receipt['support_image_ids']==packet['support_image_ids'] and len(set(receipt['support_image_ids']))==31 and
        receipt['support_preflight_sha256']==packet['support_preflight_sha256'],'wrong support31 identity')
    require(Path(receipt['adapter']['root']).resolve()==(ARM_ROOT/'training/adapter').resolve(),'wrong wide checkpoint path')
    require(file_hash(ARM_ROOT/'training/inputs.json')==receipt['inputs_sha256'],'wide training packet changed')
    return validate_source_receipt(receipt,packet)


def locked_checkpoint():
    packet=json.loads((OUTPUT/'manifest.json').read_text());path=ARM_ROOT/'training/receipt.json';receipt=json.loads(path.read_text())
    return receipt,validate_receipt(receipt,packet),file_hash(path)


def shard_records(packet,shard):
    require(type(shard) is int and 0<=shard<4,'invalid shard')
    by_id=indexed(packet['records'],'example_id');r=[by_id[eid] for eid in packet['shards'][shard]['example_ids']]
    require(len(r)==COUNTS[shard] and len({x['example_id'] for x in r})==COUNTS[shard],'exact shard count/IDs')
    return r


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
        require(all(abs(v)<=1e-5 for v in deltas.values()) and observed['A_id']==saved['target_id'] and all(observed[k]==saved[k] for k in ('rank_min','top1_id')),'wide saved/cold entry mismatch')
        publish(run/f'{stem}-cold-entry.json',dict(readout=observed,trainer=saved,deltas=deltas,tolerance=1e-5,logits_sha256=file_hash(path)))
    return 2


def require_complete_shards(terminals,packet):
    require(len(terminals)==4 and {t['shard'] for t in terminals}==set(range(4)),'fourunique receipts required')
    for t in terminals:
        i=t['shard'];require(t['status']=='completed' and t['manifest_sha256']==digest(packet) and t['gpu']==GPUS[i] and t['continuations']==COUNTS[i]
            and t['model_loads']==1 and t['score_forwards']==(2 if i==0 else 0) and t['new_tokens']<=COUNTS[i]*3084 and t['elapsed_seconds']<3600,'incomplete/wrong wide shard')
    require(len({t['training_receipt_sha256'] for t in terminals})==1,'mixed candidate receipts')


def reduce_records(cold,packet):
    controls=indexed(packet['kl10_controls'],'example_id');old=indexed(packet['records'],'example_id');result={}
    for split in ('train','guard','dev112','dev128_descriptive'):
        selected=[r for r in cold if old[r['example_id']]['split']==split or split=='dev128_descriptive' and old[r['example_id']]['split']!='train']
        result[split]=dict(images=len(selected),source=aggregate_scores([old[r['example_id']]['baseline_score'] for r in selected]),
            kl10=aggregate_scores([controls[r['example_id']]['score'] for r in selected]),wide31=aggregate_scores([r['score'] for r in selected]),
            versus={label:{t:{k:sum(len(owner_change(r['score'][t]['owners'],(old[r['example_id']]['baseline_score'] if label=='source' else controls[r['example_id']]['score'])[t]['owners'])[k]) for r in selected)
                for k in ('gained','lost','retained')} for t in ('50','60','80')} for label in ('source','kl10')},eos=sum(r['stop_reason']=='im_end' for r in selected))
    per_image={}
    for r in cold:
        eid=r['example_id'];p=old[eid];control=controls[eid]
        per_image[eid]=dict(split=p['split'],versus_source=r['owner_changes'],versus_kl10={t:owner_change(r['score'][t]['owners'],control['score'][t]['owners']) for t in ('50','60','80')},
            source=p['baseline_score'],kl10=control['score'],wide31=r['score'])
        if p['split']=='train':
            owner=p['bridge_case']['owner'];per_image[eid].update(training_readout=r['training_readout'],target_boxes={label:target_boxes(parsed,scored,owner)
                for label,parsed,scored in [('source',p['baseline'],p['baseline_score']),('kl10',control['parsed'],control['score']),('wide31',r['parsed'],r['score'])]})
    return dict(schema='selective_preservation_wide.eval_reduction.v1',images=130,splits=result,per_image=per_image,
        limitation='All panels are exposed development; combineddev128 includes earlierselectedguard16. Outcome read, not a promotion verdict.')


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
        require(terminal['continuations']==len(records),'complete assigned wide shard required');terminal['status']='completed'
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
