"""Frozen positive7/support47 candidate read on the existing384-image union."""
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
from .entrance_ce_eval import consume as native_consume,validate_natural,aggregate_scores,owner_change
from .selective_preservation_eval import target_boxes
from .selective_preservation_strong_eval import ARM_ROOT as STRONG_ROOT,validate_admission as validate_accepted_path
from .branch_bridge import summarize_logits
from .route_access import CONFIG,checkpoint_config,publish
from .round1_realization import ROOT
from probes.source_rweak_row_cross.owner_row_robustness import native_record
from probes.source_rweak_row_cross.run import build_requests

ARM_ROOT=ROOT/'2026-09-10-selective-owner-learning-autonomous/positive7-support47-81'
TRAINING=ARM_ROOT/'training';OUTPUT=ARM_ROOT/'evaluation'
EXPANDED=STRONG_ROOT/'expanded-train254';BRANCH=ARM_ROOT.parent/'positive-branch-expansion'
ADMISSION='ddp_path_reuse_positive7_support47_v1'
GPUS=tuple(range(8));COUNTS=(48,)*8
TABLE={368:(97,139,102,134,'2022537'),7116:(22,46,27,41,'181378'),252411:(1,31,11,21,'1676022'),
       465695:(1,29,9,21,'1612233'),529411:(1,21,11,11,'2147505'),538814:(4,10,9,5,'1121312'),540567:(1,49,10,40,'1491504')}
SEEDS={252411:2026090602,465695:2026090601,529411:2026090603,538814:2026090602,540567:2026090601}
BRANCH_INPUT_SHA='710c21601156a6e04a3ad195d71f824203c0a2f7eb3cf080dd0b7da79180eb4f'
BRANCH_REDUCTION_SHA='4bc95c3d736dab4d2614d087c67c24a36477f621ea12a14605fb180d6c712728'
PRIOR_SHA='037f1af3561e5522837364c41266af135e20c3d80579767092c63b761d6821ba'


def validate_mask(case,trajectory):
    iid=int(case['image_id']);require(iid in TABLE,'unadmitted positive')
    index,length,end,count,owner=TABLE[iid];ids=trajectory['action_ids'];positions=list(range(index))+list(range(end,length))
    require(case['owner_id']==owner and case['action_index']==index and case['state_ids']==case['prefix_token_ids']==ids[:index] and
        case['target_token_id']==ids[index] and len(ids)==length and ids[-1]==151645 and ids[end-1]==151649,'positive native entrance/row identity')
    require(trajectory['preservation_positions']==positions and trajectory['excluded_positions']==list(range(index,end)) and
        trajectory['suffix_start']==end and len(positions)==count and length-1 in positions,'positive exact CE/KL mask identity')


def frozen_sources():
    packet=json.loads((EXPANDED/'manifest.json').read_text());records=packet['records']+packet['prior_records'];controls=json.loads((EXPANDED/'consumer.json').read_text())+packet['prior_consumer']
    old=json.loads((STRONG_ROOT/'training/inputs.json').read_text());support=sorted(int(i) for i in old['support_image_ids'] if int(i)!=529411)
    by_id=indexed(records,'example_id');actual=indexed(controls,'example_id');require(len(by_id)==len(actual)==384 and set(by_id)==set(actual),'original384 identity')
    train={r['image_id'] for r in packet['records']}|{368,7116};dev={r['image_id'] for r in packet['prior_records'] if r['split']!='train'}
    require(len(train)==256 and len(dev)==128 and not train&dev and len(support)==47 and not set(support)&set(TABLE),'four disjoint support/positive strata')
    ordered=[]
    for r in sorted(records,key=lambda x:x['image_id']):
        iid=r['image_id'];split='positive7' if iid in TABLE else 'support47' if iid in support else 'remaining202' if iid in train else 'dev128'
        ordered.append(dict(r,split=split))
    require({s:sum(r['split']==s for r in ordered) for s in ('positive7','support47','remaining202','dev128')}==dict(positive7=7,support47=47,remaining202=202,dev128=128),'frozen384 strata')
    return ordered,[actual[r['example_id']] for r in ordered],old,support,packet


def prepare():
    require(not OUTPUT.exists(),'occupied seven evaluation output')
    records,controls,old,support,expanded=frozen_sources();by_id=indexed(records,'example_id')
    require(file_hash(BRANCH/'inputs.json')==BRANCH_INPUT_SHA and file_hash(BRANCH/'reduction.json')==BRANCH_REDUCTION_SHA,'admitted conditional evidence changed')
    branch=json.loads((BRANCH/'inputs.json').read_text());reduction=json.loads((BRANCH/'reduction.json').read_text())
    cases=copy.deepcopy(old['cases']);trajectories=copy.deepcopy(old['trajectories']);conditional_paths=[]
    for iid,seed in SEEDS.items():
        v=next(v for v in branch['variants'] if int(v['case']['image_id'])==iid and v['seed']==seed)
        c=v['case'];b=v['boundary'];cid=c['case_id'];path=BRANCH/f'rows/{iid}-{seed}.json';r=json.loads(path.read_text());conditional_paths.append(path)
        require(reduction['prospective_selection_by_lowest_seed'][cid]==v['variant_id'],'nonadmitted conditional variant')
        index,length,end,count,owner=TABLE[iid];ids=r['action_ids'];baseline=by_id[c['example_id']]
        require(ids[:index]==v['source_ids'][:index]==baseline['baseline_ids'][:index] and ids[index]==b['forced_ids'][0] and
            b['entry_index']==index and c['group']['prompt_token_ids']==baseline['prompt_token_ids'],'conditional/Source state identity')
        entry=dict(A_id=ids[index],B_id=v['source_ids'][index],state_ids=ids[:index],action_index=index)
        cases.append(dict(case_id=cid,example_id=c['example_id'],image_id=str(iid),owner_id=owner,state_ids=ids[:index],prefix_token_ids=ids[:index],
            prompt_token_ids=baseline['prompt_token_ids'],target_token_id=ids[index],action_index=index,entrance=entry,group=c['group']))
        trajectories[cid]=dict(action_ids=ids,preservation_positions=list(range(index))+list(range(end,length)),excluded_positions=list(range(index,end)),suffix_start=end)
    cases.sort(key=lambda c:int(c['image_id']))
    for c in cases:validate_mask(c,trajectories[c['case_id']])
    require(sum(len(t['preservation_positions']) for t in trajectories.values())==273,'local273 mask states')
    refs=[r for r in old['support_cases'] if int(r['image_id'])!=529411];require(len(refs)==47 and sum(len(r['action_ids']) for r in refs)==5838,'support47 states')
    paths=[EXPANDED/'manifest.json',EXPANDED/'consumer.json',STRONG_ROOT/'training/inputs.json',BRANCH/'inputs.json',BRANCH/'reduction.json',*conditional_paths]
    packet=dict(schema='selective_preservation_seven.eval.v1',arm='positive7-support47-81',numerical_admission=ADMISSION,records=records,strong_controls=controls,
        positive_cases=cases,trajectories=trajectories,support_image_ids=list(map(str,support)),support_references=refs,
        source_model=expanded['source_model'],config=expanded['config'],source_files={**expanded['source_files'],**{str(p):file_hash(p) for p in paths}},
        shards=[dict(shard=i,gpu=i,example_ids=[r['example_id'] for r in records[i::8]]) for i in range(8)])
    validate_packet(packet);OUTPUT.mkdir(parents=True);publish(OUTPUT/'manifest.json',packet)
    print(json.dumps(dict(status='frozen',manifest_sha256=digest(packet),images=384,positive7=[c['image_id'] for c in cases],support47=len(refs),shards=[48]*8)))


def validate_packet(packet):
    require(packet['schema']=='selective_preservation_seven.eval.v1' and packet['arm']=='positive7-support47-81' and packet['numerical_admission']==ADMISSION,'wrong seven packet')
    records,controls,old,support,expanded=frozen_sources()
    require(packet['records']==records and packet['strong_controls']==controls and packet['support_image_ids']==list(map(str,support)) and
        packet['source_model']==expanded['source_model'] and packet['config']==expanded['config'],'original384/strong/Source identity changed')
    require(packet['shards']==[dict(shard=i,gpu=i,example_ids=[r['example_id'] for r in records[i::8]]) for i in range(8)],'384 roundrobin changed')
    cases=indexed(packet['positive_cases'],'case_id');require({int(c['image_id']) for c in cases.values()}==set(TABLE) and len(cases)==7,'exact positive7 IDs')
    for c in cases.values():validate_mask(c,packet['trajectories'][c['case_id']])
    for c in old['cases']:
        require(cases[c['case_id']]==c and packet['trajectories'][c['case_id']]==old['trajectories'][c['case_id']],'original2 cases/masks changed')
    require(packet['support_references']==[r for r in old['support_cases'] if int(r['image_id'])!=529411],'support conflict/removal identity')
    for p,sha in packet['source_files'].items():require(file_hash(p)==sha,'frozen Source/conditional/control bytes changed')


def validate_receipt(receipt,packet):
    require(receipt['schema_version']=='selective_preservation_seven.training.v1' and receipt['status']=='completed' and receipt['numerical_admission']==ADMISSION,'wrong/incomplete seven receipt')
    require(receipt['updates']==81 and receipt['stop_reason']=='fixed_steps' and receipt['lambda_kl']==10. and receipt['lambda_support_kl']==100.,'wrong seven81 objective')
    require(list(map(int,receipt['positive_image_ids']))==sorted(TABLE) and receipt['support_image_ids']==packet['support_image_ids'] and
        receipt['local_KL_states']==273 and receipt['positive_trajectory_tokens']==325 and receipt['support_action_states']==5838 and receipt['total_cached_states']==6111,'wrong7/47/state accounting')
    require(Path(receipt['adapter']['root']).resolve()==(TRAINING/'adapter').resolve(),'wrong seven checkpoint path')
    require(file_hash(TRAINING/'inputs.json')==receipt['inputs_sha256'],'seven input packet changed')
    expected=indexed(packet['positive_cases'],'case_id');actual=indexed(receipt['cases'],'case_id');require(set(actual)==set(expected),'sealed positive IDs')
    for cid,c in expected.items():
        for k in ('example_id','image_id','owner_id','state_ids','prefix_token_ids','prompt_token_ids','target_token_id','action_index'):
            require(actual[cid][k]==c[k],'sealed positive entrance identity')
        for k in ('A_id','B_id','state_ids','action_index'):require(actual[cid]['entrance'][k]==c['entrance'][k],'sealed entrance metadata')
    require(set(receipt['final_scores'])==set(expected) and all(receipt['final_scores'][cid]['target_id']==c['target_token_id'] for cid,c in expected.items()),'seven final isolated score coverage')
    require(receipt['source_adapter']==packet['source_model']['current_adapter'] and receipt['source_embedding']==packet['source_model']['source_embedding'],'original Source/embedding identity')
    for identity in (receipt['adapter'],receipt['source_embedding']):
        for f in identity['files']:require(file_hash(Path(identity['root'])/f['relative_path'])==f['sha256'],'saved payload changed')
    inputs=json.loads((TRAINING/'inputs.json').read_text())
    support=indexed(inputs['support_cases'],'image_id');frozen_support=indexed(packet['support_references'],'image_id');require(set(support)==set(frozen_support),'sealed support47 IDs')
    for iid,ref in frozen_support.items():
        for key in ('action_ids','prompt_token_ids','group'):require(support[iid][key]==ref[key],'sealed support reference changed')
    for cid,c in expected.items():
        t=inputs['trajectories'][cid];validate_mask(c,t)
        for k in ('action_ids','preservation_positions','excluded_positions','suffix_start'):require(t[k]==packet['trajectories'][cid][k],'sealed actual trajectory/mask changed')
    validate_global_admission(receipt)
    return receipt['adapter']


def validate_global_admission(receipt):
    ranks=receipt['rank_terminals'];require(len(ranks)==8 and {r['rank'] for r in ranks}==set(range(8)),'exact8 complete ranks')
    forwards=positives=support=0
    for r in ranks:
        require(file_hash(r['path'])==r['sha256'],'rank terminal bytes');t=json.loads(Path(r['path']).read_text());i=r['rank']
        require(t['rank']==i and t['status']=='completed' and t['updates']==81 and t['model_forwards']==([588]+[574]*6+[410])[i],'rank terminal81/count identity')
        p=t['counters']['supervised_target_tokens'];s=t['counters']['support_KL_trajectories'];require(p==(81 if i<7 else 0) and s==(486 if i<7 else 405),'actual7CE/47support uses')
        forwards+=t['model_forwards'];positives+=p;support+=s
    require(forwards==receipt['global_model_forwards']==4442 and positives==receipt['actual_positive_uses']==567 and support==receipt['actual_support_uses']==3807,'global training counters')
    launch=receipt['launcher_exit'];require(file_hash(launch['path'])==launch['sha256'] and json.loads(Path(launch['path']).read_text())['exit_code']==0,'launcher notcompleteexit0')
    # Prior path proof is reused; current seven-image objective has its own fixture.
    for key in ('prior_path_proof','objective_fixture','two_step'):
        ref=receipt[key];require(file_hash(ref['path'])==ref['sha256'],'mechanical proof changed')
    prior=receipt['prior_path_proof'];path=STRONG_ROOT/'training/receipt.json';require(Path(prior['path']).resolve()==path.resolve() and prior['sha256']==PRIOR_SHA,'wrong accepted DDP path proof')
    validate_accepted_path(json.loads(path.read_text()))
    fixture=receipt['objective_fixture'];require(Path(fixture['path']).resolve().parent==TRAINING.resolve(),'wrong current objective proof path')
    f=json.loads(Path(fixture['path']).read_text());require(f['status']=='passed' and len(f['real_token_masks'])==7 and
        0<=f['max_gradient_weight_error']<=1e-6 and f['positive_CE_weight']==1/7 and f['local_KL_image_weight']==10/7 and f['support_KL_image_weight']==100/47 and
        all(f[k] is True for k in ('old_half_weight_rejected','state_pooling_rejected','compensation_and_clip_checks','excluded_nonentry_invariance')),'seven objective-gradient fixture failed')
    require(Path(receipt['two_step']['path']).resolve()==(TRAINING/'two-step-smoke.json').resolve(),'wrong current two-step path')
    smoke=json.loads(Path(receipt['two_step']['path']).read_text());require(smoke['status']=='passed' and smoke['updates']==[1,2] and smoke['initial_KL_zero'] is True and
        smoke['initial_reference_count']==54 and smoke['frozen_bytes_unchanged'] is True and smoke['reduced_gradient_adapter_optimizer_rank_identity'] is True and
        set(smoke['step2_max_KL_by_rank'])==set(map(str,range(8))) and any(v>1e-8 for v in smoke['step2_max_KL_by_rank'].values()),'current two-step54 admission')


def locked_checkpoint():
    p=json.loads((OUTPUT/'manifest.json').read_text());path=TRAINING/'receipt.json';r=json.loads(path.read_text());return r,validate_receipt(r,p),file_hash(path)


def shard_records(packet,shard):
    require(type(shard) is int and 0<=shard<8,'invalid shard');by_id=indexed(packet['records'],'example_id');r=[by_id[i] for i in packet['shards'][shard]['example_ids']]
    require(len(r)==48 and len({x['example_id'] for x in r})==48,'exact48 shard IDs');return r


def consume(path,packet,tokenizer,expected):
    result=native_consume(path,packet,tokenizer,expected);cases=indexed(packet['positive_cases'],'example_id')
    for r in result:
        if r['example_id'] not in cases:continue
        c=cases[r['example_id']];ids=r['action_ids'];i=c['action_index'];reached=ids[:i]==c['state_ids'];owner=c['owner_id']
        r['positive_readout']=dict(owner_id=owner,target_global={t:owner in r['score'][t]['owners'] for t in ('50','60','80')},
            target_direct={t:bool(r['direct_incidence'][t][owner]) for t in ('50','60','80')},exact_entrance_reached=reached,
            target_token_chosen_at_exact_entrance=reached and len(ids)>i and ids[i]==c['target_token_id'],
            earlier_token_differences=[dict(index=j,source_id=a,candidate_id=b) for j,(a,b) in enumerate(zip(c['state_ids'],ids[:i])) if a!=b],
            limitation='Exact token state mismatch does not imply semantic-state absence; natural target matching is decisive.')
    return result


def reduce_records(cold,packet):
    old=indexed(packet['records'],'example_id');strong=indexed(packet['strong_controls'],'example_id');cases=indexed(packet['positive_cases'],'example_id');panels={}
    for label in ('positive7','support47','remaining202','dev128','outside330','train256_descriptive','union384_descriptive'):
        selected=[r for r in cold if old[r['example_id']]['split']==label or label=='outside330' and old[r['example_id']]['split'] in ('remaining202','dev128') or
            label=='train256_descriptive' and old[r['example_id']]['split']!='dev128' or label=='union384_descriptive']
        panels[label]=dict(images=len(selected),source=aggregate_scores([old[r['example_id']]['baseline_score'] for r in selected]),strong100=aggregate_scores([strong[r['example_id']]['score'] for r in selected]),
            seven=aggregate_scores([r['score'] for r in selected]),versus={name:{t:{k:sum(len(owner_change(r['score'][t]['owners'],(old[r['example_id']]['baseline_score'] if name=='source' else strong[r['example_id']]['score'])[t]['owners'])[k]) for r in selected)
                for k in ('gained','lost','retained')} for t in ('50','60','80')} for name in ('source','strong100')},eos=sum(r['stop_reason']=='im_end' for r in selected))
    per_image={}
    for r in cold:
        eid=r['example_id'];per_image[eid]=dict(stratum=old[eid]['split'],source=old[eid]['baseline_score'],strong100=strong[eid]['score'],seven=r['score'],
            versus_source=r['owner_changes'],versus_strong100={t:owner_change(r['score'][t]['owners'],strong[eid]['score'][t]['owners']) for t in ('50','60','80')})
        if eid in cases:
            owner=cases[eid]['owner_id'];per_image[eid].update(positive_readout=r['positive_readout'],target_boxes={label:target_boxes(parsed,scored,owner) for label,parsed,scored in
                [('source',old[eid]['baseline'],old[eid]['baseline_score']),('strong100',strong[eid]['parsed'],strong[eid]['score']),('seven',r['parsed'],r['score'])]})
    return dict(schema='selective_preservation_seven.eval_reduction.v1',images=384,panels=panels,per_image=per_image,
        limitation='Fixed exposed acquisition/development384; outside330 excludes this update support, not original history. No confirmation or new GT.')


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
    projection=OUTPUT/'visualization-input';projection.mkdir(exist_ok=False);positive=[r for r in cold if r['split']=='positive7']
    for name in ('gt_vs_pred.jsonl','gt_vs_pred_scored.jsonl'):
        with (projection/name).open('x') as stream:
            for r in positive:stream.write(json.dumps(r['parsed'])+'\n')
    publish(projection/'provenance.json',dict(source=str(OUTPUT/'consumer.json'),scope='Unchanged native geometry projection, no invented likelihood scores.'))
    require(not (OUTPUT/'visualizations').exists(),'occupied visual output');result=render_gt_vs_prediction(projection,OUTPUT/'visualizations',duplicate_iou_threshold=.95)
    require(len(result.image_paths)==7,'positive7 visualization coverage')


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
        require(len(list((OUTPUT/'visualizations').glob('*.png')))==7,'positive7 visual coverage')
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
