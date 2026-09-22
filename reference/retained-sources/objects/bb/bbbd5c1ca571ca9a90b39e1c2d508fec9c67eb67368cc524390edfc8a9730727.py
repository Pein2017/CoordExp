"""Support50 targeted coverage: unchanged384 natural read with visible cap identities."""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import resource
import signal
import time
import traceback

from .candidate_opportunity import digest,file_hash,indexed,require,rows
from .entrance_ce_eval import validate_natural,aggregate_scores,owner_change
from .selective_preservation_seven_eval import TABLE,validate_mask,consume,validate_global_admission as validate_prior_path
from .selective_preservation_eval import target_boxes
from .branch_bridge import summarize_logits
from .route_access import CONFIG,checkpoint_config,publish
from .round1_realization import ROOT,UPDATE_ROOT
from probes.source_rweak_row_cross.owner_row_robustness import native_record
from probes.source_rweak_row_cross.run import build_requests

ARM_ROOT=ROOT/'2026-09-10-selective-owner-learning-autonomous/positive7-support50-81'
TRAINING=ARM_ROOT/'training';OUTPUT=ARM_ROOT/'evaluation'
OLD_ROOT=ARM_ROOT.parent/'positive7-support47-81'
ADMISSION='ddp_path_reuse_positive7_support50_v1'
PRIOR_SHA='7eee80e9684bd7e094e04371b11fcc00306c12e42476cc6a1876459d51fe7652'
TARGETED={73843:(10,1372),360573:(95,1415),545632:(81,1365)}
GPUS=tuple(range(8));COUNTS=(48,)*8


def expected_sources():
    old=json.loads((OLD_ROOT/'evaluation/manifest.json').read_text());controls=json.loads((OLD_ROOT/'evaluation/consumer.json').read_text())
    support=sorted(set(map(int,old['support_image_ids']))|set(TARGETED));require(len(support)==50 and not set(support)&set(TABLE),'support50/positive overlap')
    records=[dict(r,split='support50' if r['image_id'] in support else 'remaining199' if r['split']=='remaining202' else r['split']) for r in old['records']]
    require({s:sum(r['split']==s for r in records) for s in ('positive7','support50','remaining199','dev128')}==dict(positive7=7,support50=50,remaining199=199,dev128=128),'exact disjoint384 strata')
    return old,records,controls,list(map(str,support))


def prepare():
    require(not OUTPUT.exists(),'occupied stable evaluation output')
    old,records,controls,support=expected_sources();refs=copy.deepcopy(old['support_references']);by_id={r['image_id']:r for r in records}
    plan_path=UPDATE_ROOT/'round-1/plan.json';groups=indexed(json.loads(plan_path.read_text())['population']['groups'],'example_id')
    for iid,(length,total) in TARGETED.items():
        r=by_id[iid];ids=r['baseline_ids'];require(len(ids)==length and ids[-1]==151645 and r['baseline']['decode_stop_reason']=='im_end' and
            r['baseline']['dropped_prediction_count']==0 and len(r['prompt_token_ids'])+length==total,'targeted Source clean-EOS trajectory changed')
        group=groups[r['example_id']];require(group['prompt_token_ids']==r['prompt_token_ids'],'targeted Source prompt/group identity')
        refs.append(dict(action_ids=ids,example_id=r['example_id'],group=group,image_id=str(iid),prompt_token_ids=r['prompt_token_ids']))
    refs.sort(key=lambda r:int(r['image_id']));require(len(refs)==50 and sum(len(r['action_ids']) for r in refs)==6024,'support50 action-state count')
    packet=dict(old);packet.update(schema='selective_preservation_stable.eval.v1',arm='positive7-support50-81',numerical_admission=ADMISSION,
        records=records,seven_controls=controls,support_image_ids=support,support_references=refs,added_support_image_ids=list(map(str,TARGETED)),
        old_outside330_ids=[r['example_id'] for r in old['records'] if r['split'] in ('remaining202','dev128')],
        targeted_stop_ids=[by_id[i]['example_id'] for i in TARGETED],previous_manifest_sha256=file_hash(OLD_ROOT/'evaluation/manifest.json'))
    packet['source_files']=dict(old['source_files'],**{str(p):file_hash(p) for p in [OLD_ROOT/'evaluation/manifest.json',OLD_ROOT/'evaluation/consumer.json',plan_path]})
    validate_packet(packet);OUTPUT.mkdir(parents=True);publish(OUTPUT/'manifest.json',packet)
    print(json.dumps(dict(status='frozen',manifest_sha256=digest(packet),images=384,strata=[7,50,199,128],added_support_image_ids=packet['added_support_image_ids'],shards=[48]*8)))


def validate_packet(packet):
    require(packet['schema']=='selective_preservation_stable.eval.v1' and packet['arm']=='positive7-support50-81' and packet['numerical_admission']==ADMISSION,'wrong stable packet')
    old,records,controls,support=expected_sources();require(packet['records']==records and packet['seven_controls']==controls and packet['support_image_ids']==support,'384/previous control/strata identity changed')
    require(file_hash(OLD_ROOT/'evaluation/manifest.json')==packet['previous_manifest_sha256'],'previous384 manifest changed')
    for key in ('positive_cases','trajectories','strong_controls','shards','source_model','config'):require(packet[key]==old[key],'unchanged positives/masks/controls/native policy changed')
    require(packet['added_support_image_ids']==list(map(str,TARGETED)) and packet['old_outside330_ids']==[r['example_id'] for r in old['records'] if r['split'] in ('remaining202','dev128')],'targeted cases/old outside panel changed')
    require(packet['targeted_stop_ids']==[f'coco2017_train_{iid:012d}' for iid in TARGETED],'targeted stop case reporting changed')
    refs=indexed(packet['support_references'],'image_id');prior=indexed(old['support_references'],'image_id');require(set(refs)==set(support) and len(refs)==50,'exact support50 IDs')
    for iid,r in prior.items():require(refs[iid]==r,'old47 support modified')
    baseline={r['image_id']:r for r in records}
    groups=indexed(json.loads((UPDATE_ROOT/'round-1/plan.json').read_text())['population']['groups'],'example_id')
    for iid,(length,total) in TARGETED.items():
        ref=refs[str(iid)];r=baseline[iid];require(ref['action_ids']==r['baseline_ids'] and ref['prompt_token_ids']==r['prompt_token_ids'] and
            ref['group']==groups[r['example_id']] and len(ref['action_ids'])==length and ref['action_ids'][-1]==151645 and len(ref['prompt_token_ids'])+length==total,'new reference is not exact Source EOS trajectory')
    require(sum(len(r['action_ids']) for r in refs.values())==6024,'support6024 states')
    for p,sha in packet['source_files'].items():require(file_hash(p)==sha,'frozen Source/control/provenance changed')


def validate_receipt(receipt,packet):
    require(receipt['schema_version']=='selective_preservation_stable.training.v1' and receipt['status']=='completed' and receipt['numerical_admission']==ADMISSION,'wrong/incomplete stable receipt')
    require(receipt['updates']==81 and receipt['stop_reason']=='fixed_steps' and receipt['lambda_kl']==10. and receipt['lambda_support_kl']==100.,'wrong fixed81 objective')
    require(list(map(int,receipt['positive_image_ids']))==sorted(TABLE) and receipt['support_image_ids']==packet['support_image_ids'] and
        receipt['local_KL_states']==273 and receipt['positive_trajectory_tokens']==325 and receipt['support_action_states']==6024 and receipt['total_cached_states']==6297,'wrong7/50/state identity')
    require(Path(receipt['adapter']['root']).resolve()==(TRAINING/'adapter').resolve() and file_hash(TRAINING/'inputs.json')==receipt['inputs_sha256'],'wrong candidate/input packet')
    expected=indexed(packet['positive_cases'],'case_id');actual=indexed(receipt['cases'],'case_id');require(set(actual)==set(expected),'sealed seven IDs')
    for cid,c in expected.items():
        for k in ('example_id','image_id','owner_id','state_ids','prefix_token_ids','prompt_token_ids','target_token_id','action_index'):require(actual[cid][k]==c[k],'sealed entrance identity')
        for k in ('A_id','B_id','state_ids','action_index'):require(actual[cid]['entrance'][k]==c['entrance'][k],'sealed entrance metadata')
    require(set(receipt['final_scores'])==set(expected) and all(receipt['final_scores'][cid]['target_id']==c['target_token_id'] for cid,c in expected.items()),'seven terminal score identity')
    require(receipt['source_adapter']==packet['source_model']['current_adapter'] and receipt['source_embedding']==packet['source_model']['source_embedding'],'Source/embedding identity')
    for identity in (receipt['adapter'],receipt['source_embedding']):
        for f in identity['files']:require(file_hash(Path(identity['root'])/f['relative_path'])==f['sha256'],'saved payload bytes changed')
    inputs=json.loads((TRAINING/'inputs.json').read_text());support=indexed(inputs['support_cases'],'image_id');frozen=indexed(packet['support_references'],'image_id');require(set(support)==set(frozen),'sealed support50 IDs')
    for iid,r in frozen.items():
        for k in ('action_ids','prompt_token_ids','group'):require(support[iid][k]==r[k],'sealed support Source trajectory changed')
    for cid,c in expected.items():
        t=inputs['trajectories'][cid];validate_mask(c,t)
        for k in ('action_ids','preservation_positions','excluded_positions','suffix_start'):require(t[k]==packet['trajectories'][cid][k],'positive mask changed')
    validate_global_admission(receipt);return receipt['adapter']


def validate_global_admission(receipt):
    ranks=receipt['rank_terminals'];require(len(ranks)==8 and {r['rank'] for r in ranks}==set(range(8)),'exact8 complete ranks')
    forwards=positives=support=0
    for r in ranks:
        require(file_hash(r['path'])==r['sha256'],'rank receipt bytes');t=json.loads(Path(r['path']).read_text());i=r['rank']
        require(t['rank']==i and t['status']=='completed' and t['updates']==81 and t['model_forwards']==[670,656,574,574,574,574,574,492][i],'rank81/forward identity')
        p=t['counters']['supervised_target_tokens'];s=t['counters']['support_KL_trajectories'];require(p==(81 if i<7 else 0) and s==(567 if i<2 else 486),'actual7CE/50support uses')
        forwards+=t['model_forwards'];positives+=p;support+=s
    require(forwards==receipt['global_model_forwards']==4688 and positives==receipt['actual_positive_uses']==567 and support==receipt['actual_support_uses']==4050,'global training counters')
    launch=receipt['launcher_exit'];require(file_hash(launch['path'])==launch['sha256'] and json.loads(Path(launch['path']).read_text())['exit_code']==0,'launcher notexit0')
    for key in ('prior_path_proof','objective_fixture','two_step'):
        r=receipt[key];require(file_hash(r['path'])==r['sha256'],'mechanical proof changed')
    prior=receipt['prior_path_proof'];path=OLD_ROOT/'training/receipt.json';require(Path(prior['path']).resolve()==path.resolve() and prior['sha256']==PRIOR_SHA,'wrong accepted DDP path');validate_prior_path(json.loads(path.read_text()))
    fixture=receipt['objective_fixture'];require(Path(fixture['path']).resolve().parent==TRAINING.resolve(),'wrong current fixture path');f=json.loads(Path(fixture['path']).read_text())
    require(f['status']=='passed' and f['positive_and_local_unchanged'] is True and f['support_denominator']==50 and f['support_KL_image_weight']==2. and
        f['old47_denominator_rejected'] is True and f['new_Source_EOS_references_verified'] is True and f['compensation_and_clip_checks'] is True and 0<=f['max_gradient_weight_error']<=1e-6,'support50 objective/reference fixture failed')
    require(Path(receipt['two_step']['path']).resolve()==(TRAINING/'two-step-smoke.json').resolve(),'wrong current two-step path');s=json.loads(Path(receipt['two_step']['path']).read_text())
    require(s['status']=='passed' and s['updates']==[1,2] and s['initial_KL_zero'] is True and s['initial_reference_count']==57 and s['frozen_bytes_unchanged'] is True and
        s['reduced_gradient_adapter_optimizer_rank_identity'] is True and set(s['step2_max_KL_by_rank'])==set(map(str,range(8))) and any(v>1e-8 for v in s['step2_max_KL_by_rank'].values()),'current two-step57 admission')


def locked_checkpoint():
    p=json.loads((OUTPUT/'manifest.json').read_text());path=TRAINING/'receipt.json';r=json.loads(path.read_text());return r,validate_receipt(r,p),file_hash(path)


def shard_records(packet,shard):
    require(type(shard) is int and 0<=shard<8,'invalid shard');by_id=indexed(packet['records'],'example_id');r=[by_id[i] for i in packet['shards'][shard]['example_ids']]
    require(len(r)==48 and len({x['example_id'] for x in r})==48,'exact48 shard IDs');return r


def reduce_records(cold,packet):
    old=indexed(packet['records'],'example_id');strong=indexed(packet['strong_controls'],'example_id');seven=indexed(packet['seven_controls'],'example_id');cases=indexed(packet['positive_cases'],'example_id')
    outside330=set(packet['old_outside330_ids']);targeted=set(packet['targeted_stop_ids']);panels={}
    for label in ('positive7','support50','remaining199','dev128','outside327','old_outside330','targeted_stop3','train256_descriptive','union384_descriptive'):
        selected=[r for r in cold if old[r['example_id']]['split']==label or label=='outside327' and old[r['example_id']]['split'] in ('remaining199','dev128') or
            label=='old_outside330' and r['example_id'] in outside330 or label=='targeted_stop3' and r['example_id'] in targeted or
            label=='train256_descriptive' and old[r['example_id']]['split']!='dev128' or label=='union384_descriptive']
        panels[label]=dict(images=len(selected),source=aggregate_scores([old[r['example_id']]['baseline_score'] for r in selected]),strong100=aggregate_scores([strong[r['example_id']]['score'] for r in selected]),
            seven81=aggregate_scores([seven[r['example_id']]['score'] for r in selected]),stable=aggregate_scores([r['score'] for r in selected]),
            versus={name:{t:{k:sum(len(owner_change(r['score'][t]['owners'],(old[r['example_id']]['baseline_score'] if name=='source' else (strong if name=='strong100' else seven)[r['example_id']]['score'])[t]['owners'])[k]) for r in selected)
                for k in ('gained','lost','retained')} for t in ('50','60','80')} for name in ('source','strong100','seven81')},eos=sum(r['stop_reason']=='im_end' for r in selected))
    per_image={}
    for r in cold:
        eid=r['example_id'];per_image[eid]=dict(stratum=old[eid]['split'],source=old[eid]['baseline_score'],strong100=strong[eid]['score'],seven81=seven[eid]['score'],stable=r['score'],
            versus_source=r['owner_changes'],versus_seven81={t:owner_change(r['score'][t]['owners'],seven[eid]['score'][t]['owners']) for t in ('50','60','80')})
        if eid in cases:
            owner=cases[eid]['owner_id'];per_image[eid].update(positive_readout=r['positive_readout'],target_boxes={name:target_boxes(parsed,scored,owner) for name,parsed,scored in
                [('source',old[eid]['baseline'],old[eid]['baseline_score']),('strong100',strong[eid]['parsed'],strong[eid]['score']),('seven81',seven[eid]['parsed'],seven[eid]['score']),('stable',r['parsed'],r['score'])]})
    caps=dict(source=sorted(eid for eid,r in old.items() if r['baseline_score']['cap']),strong100=sorted(eid for eid,r in strong.items() if r['score']['cap']),
        seven81=sorted(eid for eid,r in seven.items() if r['score']['cap']),stable=sorted(r['example_id'] for r in cold if r['score']['cap']))
    return dict(schema='selective_preservation_stable.eval_reduction.v1',images=384,panels=panels,per_image=per_image,cap_id_sets=caps,
        cap_changes=dict(new_vs_source=sorted(set(caps['stable'])-set(caps['source'])),new_vs_seven81=sorted(set(caps['stable'])-set(caps['seven81'])),resolved_vs_seven81=sorted(set(caps['seven81'])-set(caps['stable']))),
        targeted_stop_cases={eid:per_image[eid] for eid in sorted(targeted)},
        limitation='Adaptive3 Source-reference addition from exposed failures; old outside330 and all384 remain unfiltered. No heldout/confirmation or new positive labels.')
def score_terminal_entries(qwen,packet,receipt,run):
    import numpy as np
    import torch
    from src.qwen.native import prepare_native_inputs,prepare_replay
    records=indexed(packet['records'],'example_id')
    for c in packet['positive_cases']:
        frozen=records[c['example_id']];request,_=build_requests(qwen,packet['config'],[frozen['case']])
        batch=prepare_native_inputs(qwen.processor,request,device=torch.device('cuda:0'),record_media_identity=True)
        require(list(batch.prompt_token_ids[0])==c['prompt_token_ids'] and batch.media_sha256[0]==frozen['case']['image_plan']['executed_media_sha256'],'cold entry media/prompt')
        ent=c['entrance'];cid=c['case_id'];stem=cid.replace(':','_')
        with torch.inference_mode():
            replay=prepare_replay(qwen.model,batch.inputs,prompt_token_ids=c['prompt_token_ids'],continuation_token_ids=c['state_ids']+[c['target_token_id']])
            logits=replay.aligned_logits(qwen.model(**replay.inputs).logits)
            require(logits.shape[0]==c['action_index']+1 and int(replay.target_ids[-1])==c['target_token_id'],'cold entry alignment')
            values=logits[-1].detach().float().cpu().numpy().copy();del logits,replay
        path=run/f'{stem}-cold-entry-logits.npy'
        with path.open('xb') as stream:np.save(stream,values,allow_pickle=False)
        observed=summarize_logits(np.load(path,allow_pickle=False),ent);saved=receipt['final_scores'][cid]
        deltas={k:observed[k]-saved[k] for k in ('logprob','A_vs_best_other_margin')}
        require(all(abs(v)<=1e-5 for v in deltas.values()) and observed['A_id']==saved['target_id'] and all(observed[k]==saved[k] for k in ('rank_min','top1_id')),'saved/cold seven entry mismatch')
        publish(run/f'{stem}-cold-entry.json',dict(readout=observed,trainer=saved,deltas=deltas,tolerance=1e-5,logits_sha256=file_hash(path)))
    return 7

def require_complete_shards(terminals,packet):
    require(len(terminals)==8 and {t['shard'] for t in terminals}==set(range(8)),'eightunique complete evalreceipts required')
    for t in terminals:
        i=t['shard'];require(t['status']=='completed' and t['manifest_sha256']==digest(packet) and t['gpu']==i and t['continuations']==48 and t['model_loads']==1 and
            t['score_forwards']==(7 if i==0 else 0) and t['new_tokens']<=148032 and t['elapsed_seconds']<3600,'incomplete/wrong seven evalshard')
    require(len({t['training_receipt_sha256'] for t in terminals})==1,'mixed candidate receipts')

def render_positives(cold):
    from src.vis import render_gt_vs_prediction
    projection=OUTPUT/'visualization-input';projection.mkdir(exist_ok=False);targeted=set(json.loads((OUTPUT/'manifest.json').read_text())['targeted_stop_ids'])
    positive=[r for r in cold if r['split']=='positive7' or r['example_id'] in targeted]
    for name in ('gt_vs_pred.jsonl','gt_vs_pred_scored.jsonl'):
        with (projection/name).open('x') as stream:
            for r in positive:stream.write(json.dumps(r['parsed'])+'\n')
    publish(projection/'provenance.json',dict(source=str(OUTPUT/'consumer.json'),scope='Unchanged native geometry projection, no invented likelihood scores.'))
    require(not (OUTPUT/'visualizations').exists(),'occupied visual output');result=render_gt_vs_prediction(projection,OUTPUT/'visualizations',duplicate_iou_threshold=.95)
    require(len(result.image_paths)==10,'positive7 visualization coverage')

def merge(verify_only=False):
    import numpy as np
    from tokenizers import Tokenizer
    packet=json.loads((OUTPUT/'manifest.json').read_text());validate_packet(packet);training,_,sha=locked_checkpoint()
    terminals=[json.loads((OUTPUT/f'shard-{i}/terminal.json').read_text()) for i in range(8)];require_complete_shards(terminals,packet)
    exits=json.loads((OUTPUT/'process-exits.json').read_text())['results'];require(len(exits)==8 and {r['shard'] for r in exits}==set(range(8)) and all(r['exit_code']==0 for r in exits),'eight complete exit0 required')
    require(all(t['training_receipt_sha256']==sha for t in terminals),'sealed receipt changed')
    tokenizer=Tokenizer.from_file(packet['source_model']['base_model_path']+'/tokenizer.json');collected=[]
    for i in range(8):
        run=OUTPUT/f'shard-{i}';cold=consume(run/'rows.jsonl',packet,tokenizer,[r['example_id'] for r in shard_records(packet,i)])
        require(cold==json.loads((run/'consumer.json').read_text()) and sum(len(r['action_ids']) for r in cold)==terminals[i]['new_tokens'],'shard native consumer/counter mismatch');collected.extend(cold)
    by_id=indexed(collected,'example_id');require(set(by_id)=={r['example_id'] for r in packet['records']} and len(by_id)==384,'exact merged384 IDs')
    ordered=[by_id[r['example_id']] for r in packet['records']];reduced=reduce_records(ordered,packet)
    for c in packet['positive_cases']:
        cid=c['case_id'];stem=cid.replace(':','_');run=OUTPUT/'shard-0';saved=json.loads((run/f'{stem}-cold-entry.json').read_text());path=run/f'{stem}-cold-entry-logits.npy'
        require(file_hash(path)==saved['logits_sha256'] and summarize_logits(np.load(path,allow_pickle=False),c['entrance'])==saved['readout'] and saved['trainer']==training['final_scores'][cid],'cold seven entry arithmetic/identity')
    if verify_only:
        require(ordered==json.loads((OUTPUT/'consumer.json').read_text()) and reduced==json.loads((OUTPUT/'reduction.json').read_text()),'merged CPU consumer/reduction mismatch')
        require(len(list((OUTPUT/'visualizations').glob('*.png')))==10,'positive7 visual coverage')
    else:
        publish(OUTPUT/'consumer.json',ordered);publish(OUTPUT/'reduction.json',reduced);render_positives(ordered)
        publish(OUTPUT/'resources.json',dict(model_loads=8,score_forwards=7,continuations=384,new_tokens=sum(t['new_tokens'] for t in terminals),allocated_gpu_seconds=sum(t['elapsed_seconds'] for t in terminals),shards=terminals))
    return dict(status='candidate_cpu_verified' if verify_only else 'candidate_merged',images=384,manifest_sha256=digest(packet),consumer_sha256=file_hash(OUTPUT/'consumer.json'),reduction_sha256=file_hash(OUTPUT/'reduction.json'))

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
        require(terminal['continuations']==len(records),'complete assigned seven shard required');terminal['status']='completed'
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
