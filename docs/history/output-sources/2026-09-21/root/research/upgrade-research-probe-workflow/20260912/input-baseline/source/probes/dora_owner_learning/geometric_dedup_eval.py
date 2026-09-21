"""Stable50 dedup packet and unchanged384 natural consumer; no GT training labels."""
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

from .candidate_opportunity import digest,file_hash,indexed,require,score
from .route_access import ROOT,CONFIG,checkpoint_config,publish,checked_ids
from .reward_rows import _pixel_box
from .entrance_ce_eval import aggregate_scores
from probes.source_rweak_row_cross.run import build_requests,native_record
from src.data.geometry import iou_xyxy

OUTPUT=ROOT/'2026-09-11-stable50-geometric-dedup'
ANCHOR=ROOT/'2026-09-10-selective-owner-learning-autonomous/positive7-support50-81'
ONLINE=[9813,158044,248167,274509,351017,417044,477415,502725]
CAP=3084


def overlap_counts(parsed):
    boxes=[b for p in parsed['pred'] if (b:=_pixel_box(p['bbox'])) is not None]
    return {str(t):sum(any(iou_xyxy(b,a)>t/100 for a in boxes[:i]) for i,b in enumerate(boxes)) for t in (80,90,95)}


def prepare():
    from tokenizers import Tokenizer
    from .geometric_dedup import trajectory_layout
    from .selective_preservation_stable_eval import locked_checkpoint
    require(not (OUTPUT/'inputs.json').exists(),'occupied dedup packet')
    manifest_path=ANCHOR/'evaluation/manifest.json'; consumer_path=ANCHOR/'evaluation/consumer.json'
    m=json.loads(manifest_path.read_text()); current=indexed(json.loads(consumer_path.read_text()),'example_id')
    previous_receipt,adapter,receipt_sha=locked_checkpoint()
    route_path=ROOT/'2026-09-10-fixed-witness-route-access/inputs.json'
    groups=indexed(json.loads(route_path.read_text())['plan']['population']['groups'],'example_id')
    tok=Tokenizer.from_file(m['source_model']['base_model_path']+'/tokenizer.json')
    repeat_ids=sorted(r['image_id'] for r in m['records'] if r['split']!='dev128' and current[r['example_id']]['score']['strict_repeats'])
    require(repeat_ids==ONLINE[1:],'all seven and only seven train repeat images')
    zero=sorted(r['image_id'] for r in m['records'] if r['split']=='remaining199' and
        not any(current[r['example_id']]['score'][k] for k in ('strict_repeats','parser_drops','invalid_predictions','cap')))
    require(zero[0]==ONLINE[0],'deterministic zero-repeat control changed')
    refs=[r for r in m['records'] if r['split'] in ('support50','positive7') and r['image_id']!=158044]
    refids={r['image_id'] for r in refs};require(len(refids)==56 and not refids&set(ONLINE),'disjoint online8/reference56')
    def training_case(r):
        eid=r['example_id']; cur=current[eid]; g=groups[eid]
        require(g['prompt_token_ids']==r['prompt_token_ids'],'group/Stable prompt identity')
        layout=trajectory_layout(cur['action_ids'],tok,image_width=cur['parsed']['image_width'],image_height=cur['parsed']['image_height'],row_id=eid)
        require(len(layout['duplicate_row_indices'])==cur['score']['strict_repeats'],'native pixel duplicate count differs')
        keys=('example_id','image_id','image_path','image_content_sha256','executed_media_sha256','observed_image_grid_thw','prompt_token_ids','prompt_token_ids_sha256')
        return dict(key=eid,example_id=eid,image_id=str(r['image_id']),group={k:g[k] for k in keys},case=r['case'],
            image_width=cur['parsed']['image_width'],image_height=cur['parsed']['image_height'],prompt_token_ids=r['prompt_token_ids'],
            action_ids=cur['action_ids'],action_ids_sha256=digest(cur['action_ids']),stop_reason=cur['stop_reason'],initial_layout=layout)
    old_by_image={r['image_id']:r for r in m['records']}
    online=[training_case(old_by_image[i]) for i in ONLINE]
    references=[training_case(r) for r in sorted(refs,key=lambda r:r['image_id'])]
    require(sum(len(c['initial_layout']['duplicate_row_indices']) for c in online)==495 and sum(len(c['action_ids']) for c in online)==13340,'online495/13340 identity')
    require(sum(len(c['action_ids']) for c in references)==6056 and all(not c['initial_layout']['duplicate_row_indices'] for c in references),'Stable reference count/no duplicates')
    records=[]
    for old in sorted(m['records'],key=lambda r:r['image_id']):
        cur=current[old['example_id']]; iid=old['image_id']
        split='online8' if iid in ONLINE else 'reference56' if iid in refids else 'dev128' if old['split']=='dev128' else 'remaining192'
        stable_score=score(cur['parsed'],seed=None,length=len(cur['action_ids']),stop=cur['stop_reason'])
        records.append(dict(example_id=old['example_id'],image_id=iid,split=split,case=old['case'],golden=old['baseline'],
            prompt_token_ids=old['prompt_token_ids'],stable_ids=cur['action_ids'],stable_parsed=cur['parsed'],stable_score=stable_score,
            stable_overlap_counts=overlap_counts(cur['parsed'])))
    require({s:sum(r['split']==s for r in records) for s in ('online8','reference56','remaining192','dev128')}==dict(online8=8,reference56=56,remaining192=192,dev128=128),'exact384 partition')
    protected={str(c['image_id']):str(c['owner_id']) for c in m['positive_cases']}
    require(len(protected)==7 and all(o in current[f'coco2017_train_{int(i):012d}']['score']['50']['owners'] for i,o in protected.items()),'seven current protected targets')
    model=dict(m['source_model'],current_adapter=adapter)
    paths=[manifest_path,consumer_path,ANCHOR/'training/receipt.json',route_path,CONFIG,Path(__file__),
        Path(__file__).with_name('geometric_dedup.py'),Path(__file__).with_name('geometric_dedup_train.py'),
        Path(__file__).parent/'tests/test_geometric_dedup.py',Path(__file__).parents[2]/'tests/test_geometric_dedup_train.py',
        Path(__file__).parent/'tests/test_geometric_dedup_eval.py',
        Path(model['base_model_path'])/'tokenizer.json']
    sources={str(p):file_hash(p) for p in paths}
    for identity in (adapter,model['source_embedding']):
        for f in identity['files']:
            path=Path(identity['root'])/f['relative_path']; require(file_hash(path)==f['sha256'],'anchor payload changed');sources[str(path)]=f['sha256']
    for c in online+references:
        path=Path(c['group']['image_path']);h=c['group']['image_content_sha256'];require(file_hash(path)==h,'training media changed');sources[str(path)]=h
    packet=dict(schema='stable50_geometric_dedup.inputs.v1',model=model,anchor_training_receipt_sha256=receipt_sha,config=m['config'],
        online_cases=online,reference_cases=references,eval_records=records,protected_targets=protected,source_files=sources,
        numerical_policy=dict(dtype='fp32',attention='sdpa',patch_embed_linearization='enabled',temperature=0.,repetition_penalty=1.,total_action_cap=CAP,
            activation_checkpointing='non_reentrant_language_blocks_grad_only_eval_mode'),
        optimizer=dict(lr=1e-5,betas=[.9,.999],eps=1e-8,weight_decay=0.,foreach=False),
        objective=dict(online_images=8,reference_images=56,lambda_ul=1.,lambda_online_kl=10.,lambda_reference_kl=100.,
            description='mean8(UL+10*KL_nondup)+100*mean56(KL_nondup); no positive CE or GT negative labels',
            duplicate_iou_strict=.95,epsilon=1e-8,credit='four sampled coordinate-token geometric-mean confidence'),
        limits=dict(world_size=8,updates=32,refresh_steps=[0,8,16,24],clip_gradient_norm=1.,train_seconds_per_rank=3600,
            eval_seconds_per_shard=3600,total_action_cap=CAP,online_continuations=32,train_forwards_per_rank=256,
            teacher_forwards_per_rank=11,max_model_forwards_per_rank=4*(CAP+1)+267,max_image_forwards_per_rank=271,
            decoder_checkpoint_layers=28,parity_rank=1,parity_forwards=2,
            max_model_forwards_by_rank=[4*(CAP+1)+267+(2 if i==1 else 0) for i in range(8)],
            max_image_forwards_by_rank=[271+(2 if i==1 else 0) for i in range(8)]),
        eval_shards=[[r['example_id'] for r in records[i::8]] for i in range(8)],
        training_strata='All7 train strict-repeat cases + zero9813; current Stable static56, no dev learning.',
        claim_boundary='One32-update pilot. Annotation-relative natural384 quality; no confirmation or deployment promotion.',
        technical_retry_of=dict(path=str(OUTPUT/'technical-invalid-attempt-01/inputs.json'),
            sha256=file_hash(OUTPUT/'technical-invalid-attempt-01/inputs.json'),reason='OOM before first global update; activation recomputation only'))
    OUTPUT.mkdir(parents=True,exist_ok=True);publish(OUTPUT/'inputs.json',packet)
    return dict(input_sha256=file_hash(OUTPUT/'inputs.json'),online_ids=ONLINE,reference_images=56,
        online_baseline_repeats=495,online_KL_states=sum(len(c['initial_layout']['kl_positions']) for c in online),
        reference_KL_states=sum(len(c['initial_layout']['kl_positions']) for c in references),evaluation_images=len(records))


def candidate_adapter(packet):
    training=OUTPUT/'training'; path=training/'receipt.json';receipt=json.loads(path.read_text())
    require(receipt['status']=='completed' and receipt['updates']==32,'complete fixed32 training required')
    require(receipt['inputs_sha256']==file_hash(OUTPUT/'inputs.json'),'training packet identity')
    adapter=receipt['adapter'];require(Path(adapter['root']).resolve()==(training/'adapter').resolve(),'wrong candidate adapter root')
    for f in adapter['files']:require(file_hash(Path(adapter['root'])/f['relative_path'])==f['sha256'],'saved candidate payload changed')
    require(packet['model']['current_adapter']['fingerprint']!=adapter['fingerprint'],'candidate did not change')
    return adapter,file_hash(path)


def consume(raw,frozen,tok):
    require(raw['example_id']==frozen['example_id'] and raw['prefix_ids']==[] and raw['forced_ids']==[] and raw['remaining_budget']==CAP,'unforced original-input association')
    ids=raw['action_ids'];checked_ids(ids,raw['stop_reason'])
    require(len(ids)<=CAP and tok.decode(ids,skip_special_tokens=False)==raw['text'],'natural token/text/budget')
    parsed=native_record(raw['text'],frozen['case'],frozen['golden'],raw['stop_reason'])
    require(parsed==raw['parsed'],'cold native parser differs')
    return dict(raw,score=score(parsed,seed=None,length=len(ids),stop=raw['stop_reason']),overlap_counts=overlap_counts(parsed))


def execute(shard):
    import torch
    from tokenizers import Tokenizer
    from src.config.inference import load_research_infer_config
    from src.qwen.native import prepare_native_inputs
    from src.qwen.generation import generate_continuations,NativeGenerationPolicy
    from .runtime import load_policy
    p=json.loads((OUTPUT/'inputs.json').read_text());require(type(shard) is int and 0<=shard<8,'valid shard')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')==str(shard) and torch.cuda.device_count()==1,'assigned single GPU')
    for path,sha in p['source_files'].items():require(file_hash(path)==sha,'frozen source changed: '+path)
    adapter,receipt_sha=candidate_adapter(p)
    out=OUTPUT/'evaluation';out.mkdir(exist_ok=True);run=out/f'shard-{shard}';run.mkdir(exist_ok=False)
    started=time.monotonic();t=dict(status='running',shard=shard,pid=os.getpid(),inputs_sha256=file_hash(OUTPUT/'inputs.json'),
        training_receipt_sha256=receipt_sha,model_loads=0,continuations=0,new_tokens=0,model_forwards=0,image_forwards=0)
    publish(run/'launch.json',t)
    def expired(*_):raise TimeoutError('3600 second evaluation shard bound')
    signal.signal(signal.SIGALRM,expired);signal.alarm(3600)
    try:
        cfg=checkpoint_config(load_research_infer_config(CONFIG).config,adapter['root'])
        require(str(cfg.model.base_model)==p['model']['base_model_path'] and str(cfg.embedding_delta.path)==p['model']['source_embedding']['root'],'base/embedding identity')
        qwen,identity=load_policy(cfg,device=torch.device('cuda:0'));t['model_loads']=1
        require(identity['model_identity']['adapter']['adapter_path']==adapter['root'] and not identity['model_identity']['adapter']['merged_adapters'],'candidate model identity')
        require(identity['effective_settings']['observed_model_dtype']['parameter_dtype_names']==['torch.float32'] and identity['effective_settings']['observed_attn_implementation']=='sdpa','observed native numerics')
        publish(run/'model.json',identity);publish(run/'config.json',cfg.model_dump(mode='json'))
        qwen.model.eval()
        for parameter in qwen.model.parameters():parameter.requires_grad_(False)
        tok=Tokenizer.from_file(p['model']['base_model_path']+'/tokenizer.json');by_id=indexed(p['eval_records'],'example_id')
        policy=NativeGenerationPolicy(temperature=0.,top_p=1.,repetition_penalty=1.,top_k=0,use_model_defaults=False)
        def model_count(*_):t['model_forwards']+=1;require(t['model_forwards']<=48*(CAP+1),'eval forward bound')
        def image_count(*_):t['image_forwards']+=1;require(t['image_forwards']<=48,'eval image bound')
        qwen.model.register_forward_pre_hook(model_count);visual=[v for n,v in qwen.model.named_modules() if n.endswith('visual')]
        require(len(visual)==1,'visual module identity');visual[0].register_forward_pre_hook(image_count)
        torch.cuda.reset_peak_memory_stats()
        with (run/'rows.jsonl').open('x') as stream:
            for eid in p['eval_shards'][shard]:
                c=by_id[eid];requests,_=build_requests(qwen,p['config'],[c['case']]);plan=c['case']['image_plan']
                batch=prepare_native_inputs(qwen.processor,requests,device=torch.device('cuda:0'),record_media_identity=True)
                require(list(batch.prompt_token_ids[0])==c['prompt_token_ids'] and batch.media_sha256[0]==plan['executed_media_sha256'] and list(batch.image_grids[0])==plan['observed_image_grid_thw'],'actual prompt/media/grid')
                with torch.inference_mode():
                    result=generate_continuations(qwen.model,batch,extensions=[[]],budgets=[CAP],eos_token_id=151645,pad_token_id=qwen.tokenizer.pad_token_id,policy=policy,trace='none')[0]
                require(result.request_id==eid,'request association');ids=list(result.token_ids);text=qwen.tokenizer.decode(ids,skip_special_tokens=False)
                raw=dict(example_id=eid,image_id=c['image_id'],split=c['split'],action_ids=ids,prefix_ids=[],forced_ids=[],remaining_budget=CAP,
                    text=text,stop_reason=result.stop_reason,parsed=native_record(text,c['case'],c['golden'],result.stop_reason))
                consume(raw,c,tok);stream.write(json.dumps(raw)+'\n');stream.flush();os.fsync(stream.fileno())
                t['continuations']+=1;t['new_tokens']+=len(ids)
        require(t['continuations']==t['image_forwards']==48,'exact48 native continuations');t['status']='completed'
    except BaseException as exc:
        t.update(status='failed',error=repr(exc),traceback=traceback.format_exc());raise
    finally:
        signal.alarm(0);t.update(elapsed_seconds=time.monotonic()-started,rss_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0)
        publish(run/'terminal.json',t)


def reduce_records(cold,packet):
    before=indexed(packet['eval_records'],'example_id');panels={}
    for label in ('online8','reference56','remaining192','dev128','train256','union384'):
        selected=[r for r in cold if r['split']==label or label=='train256' and r['split']!='dev128' or label=='union384']
        old=[before[r['example_id']]['stable_score'] for r in selected];new=[r['score'] for r in selected]
        changes={}
        for threshold in ('50','60','80'):
            pairs=[(set(a[threshold]['owners']),set(b[threshold]['owners'])) for a,b in zip(old,new)]
            changes[threshold]={k:sum(len(v) for v in sets) for k,sets in {
                'gained':[b-a for a,b in pairs],'lost':[a-b for a,b in pairs],'retained':[a&b for a,b in pairs]}.items()}
        panels[label]=dict(images=len(selected),anchor=aggregate_scores(old),candidate=aggregate_scores(new),owner_changes=changes,
            overlap_counts={name:{str(t):sum((before[r['example_id']]['stable_overlap_counts'] if name=='anchor' else r['overlap_counts'])[str(t)] for r in selected) for t in (80,90,95)} for name in ('anchor','candidate')})
    per_image=[]
    for row in cold:
        old=before[row['example_id']];changes={}
        for t in ('50','60','80'):
            a,b=set(old['stable_score'][t]['owners']),set(row['score'][t]['owners'])
            changes[t]=dict(gained=sorted(b-a),lost=sorted(a-b),retained=sorted(a&b))
        per_image.append(dict(example_id=row['example_id'],image_id=row['image_id'],split=row['split'],owner_changes=changes,
            anchor=old['stable_score'],candidate=row['score'],anchor_overlap=old['stable_overlap_counts'],candidate_overlap=row['overlap_counts']))
    caps={name:sorted(r['image_id'] for r in per_image if r[name]['cap']) for name in ('anchor','candidate')}
    byimage={str(r['image_id']):r for r in cold}
    protected={iid:{t:owner in byimage[iid]['score'][t]['owners'] for t in ('50','60','80')} for iid,owner in packet['protected_targets'].items()}
    return dict(schema='stable50_geometric_dedup.reduction.v1',panels=panels,per_image=per_image,cap_image_ids=caps,
        new_cap_images=sorted(set(caps['candidate'])-set(caps['anchor'])),resolved_cap_images=sorted(set(caps['anchor'])-set(caps['candidate'])),
        protected_targets=protected,claim='Natural exposed384 annotation-relative comparison; no held-out generalization.')


def merge(verify=False):
    from tokenizers import Tokenizer
    p=json.loads((OUTPUT/'inputs.json').read_text());out=OUTPUT/'evaluation';tok=Tokenizer.from_file(p['model']['base_model_path']+'/tokenizer.json')
    _,receipt_sha=candidate_adapter(p);exits=json.loads((out/'process-exits.json').read_text())['results']
    require(len(exits)==8 and {x['shard'] for x in exits}==set(range(8)) and all(x['exit_code']==0 for x in exits),'complete8 evaluation exits')
    by_id=indexed(p['eval_records'],'example_id');cold=[];terminals=[]
    for i in range(8):
        run=out/f'shard-{i}';t=json.loads((run/'terminal.json').read_text());terminals.append(t)
        require(t['status']=='completed' and t['shard']==i and t['training_receipt_sha256']==receipt_sha and t['inputs_sha256']==file_hash(OUTPUT/'inputs.json'),'shard provenance')
        raw=[json.loads(s) for s in (run/'rows.jsonl').read_text().splitlines()]
        require([r['example_id'] for r in raw]==p['eval_shards'][i] and t['continuations']==t['image_forwards']==48,'exact ordered shard population')
        require(sum(len(r['action_ids']) for r in raw)==t['new_tokens'],'actual token counter')
        cold.extend(consume(r,by_id[r['example_id']],tok) for r in raw)
    require(len(cold)==len({r['example_id'] for r in cold})==384,'unique exact384 consumer')
    cold.sort(key=lambda r:r['image_id']);reduction=reduce_records(cold,p)
    if verify:
        require(cold==json.loads((out/'consumer.json').read_text()) and reduction==json.loads((out/'reduction.json').read_text()),'cold merged identity')
    else:
        publish(out/'consumer.json',cold);publish(out/'reduction.json',reduction)
        publish(out/'resources.json',dict(model_loads=8,continuations=384,new_tokens=sum(t['new_tokens'] for t in terminals),allocated_gpu_seconds=sum(t['elapsed_seconds'] for t in terminals),shards=terminals))
    return dict(panels=reduction['panels'],new_cap_images=reduction['new_cap_images'],protected_targets=reduction['protected_targets'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','execute','merge','verify']);parser.add_argument('--shard',type=int)
    args=parser.parse_args()
    if args.command=='prepare':print(json.dumps(prepare()))
    elif args.command=='execute':execute(args.shard)
    else:print(json.dumps(merge(verify=args.command=='verify')))
