"""Independent frozen natural evaluation of the saved two-entrance CE candidate."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import time
import traceback

from .candidate_opportunity import digest,file_hash,indexed,require,rows,score
from .branch_bridge import OUTPUT as BRIDGE_ROOT,entrance,summarize_logits
from .route_access import CONFIG,checkpoint_config,checked_ids,first_fork,publish
from .round1_realization import ROOT,SOURCE_ROOT
from .reward_rows import _gt_objects
from probes.source_rweak_row_cross.owner_row_robustness import native_record,incidence
from probes.source_rweak_row_cross.run import build_requests

ROOT_OUTPUT=ROOT/'2026-09-10-native-entrance-ce-feasibility'
OUTPUT=ROOT_OUTPUT/'evaluation'
DEV=SOURCE_ROOT.parent/'qwen3-vl-2b-sft256-source-dev128-natural-v1'
SALT='native-entrance-ce-dev16-v1:'


def guard_selection(ids,train_ids):
    numeric=[int(i) for i in ids];require(len(numeric)==128 and len(set(numeric))==128,'dev128 image identity')
    ranked=sorted(numeric,key=lambda i:hashlib.sha256(f'{SALT}{i}'.encode()).hexdigest())
    selected=ranked[:16];require(not set(selected)&set(map(int,train_ids)),'training/guard image collision; no replacement')
    return [dict(image_id=i,rank_hash=hashlib.sha256(f'{SALT}{i}'.encode()).hexdigest()) for i in selected]


def prepare(output,*,resume_selection=False):
    import yaml
    from src.data.geometry import parse_coord_token
    from src.qwen.runtime_loading import QwenLoadOptions,load_qwen_components_from_options
    from src.qwen.native import prepare_native_inputs
    require(not output.exists() or (resume_selection and set(p.name for p in output.iterdir())=={'selection.json'}),'occupied evaluation output')
    bridge=json.loads((BRIDGE_ROOT/'inputs.json').read_text())
    configs={name:yaml.safe_load((root/'configs/resolved.yaml').read_text())['config'] for name,root in [('train',SOURCE_ROOT),('guard',DEV)]}
    inputs={name:rows(c['data']['input_jsonl']) for name,c in configs.items()}
    train_ids=[int(b['case']['image_id']) for b in bridge['bindings']]
    selection=guard_selection([r['image_id'] for r in inputs['guard']],train_ids)
    selection_receipt=dict(schema='native_entrance_ce_eval_selection.v1',train_image_ids=train_ids,
        guard16=selection,policy='numeric image ID salted SHA256 ordering; no outcome filter/backfill',
        guard_scope='historically used Source dev128; not trained by this update, not untouched confirmation')
    if output.exists():require(json.loads((output/'selection.json').read_text())==selection_receipt,'frozen selection changed')
    else:
        output.mkdir(parents=True);publish(output/'selection.json',selection_receipt)
    selected_guard={r['image_id'] for r in selection};records=[];sources={str(BRIDGE_ROOT/'inputs.json'):file_hash(BRIDGE_ROOT/'inputs.json')}
    qwen=load_qwen_components_from_options(QwenLoadOptions(base_model=configs['train']['model']['base_model'],dtype='fp32',
            attn_implementation='sdpa',patch_embed_linearization='enabled',load_model=False))
    source_identity=None
    for split,root in [('train',SOURCE_ROOT),('guard',DEV)]:
        config=configs[split];manifest=json.loads((root/'run_manifest.json').read_text())
        policy=manifest['generation_policy'];require(policy['do_sample'] is False and policy['max_new_tokens']==3084 and policy['top_p']==1 and policy['repetition_penalty']==1,'Source natural policy')
        require(config['model']['dtype']=='fp32' and config['backend']['hf']==dict(attn_implementation='sdpa',patch_embed_linearization='enabled'),'Source numerics')
        if source_identity is None:source_identity=manifest['model_identity']
        else:require(manifest['model_identity']==source_identity,'train/guard Source checkpoint mismatch')
        raw=indexed(rows(root/'gt_vs_pred.jsonl'),'row_id');images=indexed(rows(root/'image_plan.jsonl'),'row_id');prompts=indexed(manifest['prompt_trace'],'row_id')
        traces=defaultdict(list)
        for t in rows(root/'pred_token_trace.jsonl'):
            if t['trace_type']=='generated_token' and not t['is_pad']:traces[t['row_id']].append(t)
        selected=[r for r in inputs[split] if int(r['image_id']) in (set(train_ids) if split=='train' else selected_guard)]
        order={i:j for j,i in enumerate(train_ids if split=='train' else [g['image_id'] for g in selection])}
        selected.sort(key=lambda r:order[int(r['image_id'])])
        for input_row in selected:
            iid=int(input_row['image_id']);eid=f'coco2017_train_{iid:012d}';golden=raw[eid];plan=images[eid]
            native_gt=[dict(description=o['desc'],bbox=[parse_coord_token(v,field='GT') for v in o['bbox_2d']],object_id=str(o['coco_ann_id'])) for o in input_row['objects']]
            require(_gt_objects(dict(golden,gt=native_gt),row_id=eid)==_gt_objects(golden,row_id=eid) and
                [g['object_id'] for g in native_gt]==[str(g['object_id']) for g in golden['gt']],'baseline/raw GT identity')
            require(file_hash(golden['image_path'])==plan['image_content_sha256'],'baseline image bytes')
            case=dict(row_id=eid,row_index=golden['row_index'],image_path=golden['image_path'],image_width=golden['image_width'],
                image_height=golden['image_height'],input_record=input_row,image_plan=plan)
            requests,meta=build_requests(qwen,config,[case]);request=requests[0]
            batch=prepare_native_inputs(qwen.processor,[request],device='cpu',record_media_identity=True)
            prompt=list(batch.prompt_token_ids[0]);require(digest(prompt)==prompts[eid]['backend_executed_prompt_token_ids_sha256'],'baseline exact prompt identity')
            require(batch.media_sha256[0]==plan['executed_media_sha256'] and list(batch.image_grids[0])==plan['observed_image_grid_thw'],'baseline actual media/grid identity')
            trace=sorted(traces[eid],key=lambda t:t['generated_step_index']);require([t['generated_step_index'] for t in trace]==list(range(len(trace))),'baseline trace coverage')
            ids=checked_ids([t['token_id'] for t in trace],golden['decode_stop_reason'])
            require(qwen.tokenizer.decode(ids,skip_special_tokens=False)==golden['raw_decode_text'],'baseline token/text identity')
            require(native_record(golden['raw_decode_text'],case,golden,golden['decode_stop_reason'])==golden,'baseline native parser identity')
            record=dict(example_id=eid,image_id=iid,split=split,case=case,baseline=golden,baseline_ids=ids,prompt_token_ids=prompt,
                baseline_score=score(golden,seed=-1,length=len(ids),stop=golden['decode_stop_reason']))
            if split=='train':
                b=next(b for b in bridge['bindings'] if b['case']['example_id']==eid)
                require(ids==b['references']['source'] and prompt==b['group']['prompt_token_ids'],'bridge/source binding')
                record['bridge_case']=b['case'];record['entrance']=entrance(b['case'])
            records.append(record)
            del batch
        for path in [root/'run_manifest.json',root/'configs/resolved.yaml',root/'gt_vs_pred.jsonl',root/'image_plan.jsonl',root/'pred_token_trace.jsonl',Path(config['data']['input_jsonl'])]:sources[str(path)]=file_hash(path)
    require(len(records)==18 and sum(r['split']=='train' for r in records)==2,'frozen natural cohort')
    packet=dict(schema='native_entrance_ce_eval.v1',selection=json.loads((output/'selection.json').read_text()),records=records,
        configs=configs,source_identity=source_identity,source_files=sources,source_model=bridge['model'])
    publish(output/'manifest.json',packet)
    print(json.dumps(dict(status='frozen',records=18,guard_ids=[s['image_id'] for s in selection],manifest_sha256=digest(packet),manifest_file_sha256=file_hash(output/'manifest.json'))))


def validate_receipt(receipt,packet):
    require(receipt['status']=='completed','training checkpoint incomplete')
    adapter=receipt['adapter'];require(adapter['root'] and adapter['files'],'missing saved adapter identity')
    for identity in (adapter,receipt['source_embedding']):
        for f in identity['files']:
            require(file_hash(Path(identity['root'])/f['relative_path'])==f['sha256'],'checkpoint payload bytes changed')
    require(receipt['source_embedding']==packet['source_model']['source_embedding'],'original embedding identity')
    require(receipt['source_adapter']==packet['source_model']['current_adapter'],'original Source adapter identity')
    train={r['bridge_case']['case_id']:r for r in packet['records'] if r['split']=='train'}
    saved=indexed(receipt['cases'],'case_id');require(set(saved)==set(train),'training/evaluation case identity')
    require(set(receipt['final_scores'])==set(train),'training final score coverage')
    for cid,row in train.items():
        c=saved[cid];e=row['entrance']
        require(c['state_ids']==c['prefix_token_ids']==e['state_ids'] and c['target_token_id']==e['A_id'] and
                c['action_index']==e['action_index'] and c['prompt_token_ids']==row['prompt_token_ids'],'training entrance/prompt identity')
        require(receipt['final_scores'][cid]['target_id']==e['A_id'],'saved final target identity')
    require(1<=receipt['updates']<=32,'finite registered update count')
    return adapter


def validate_natural(row,frozen,tokenizer):
    require(row['example_id']==frozen['example_id'] and row['prefix_ids']==[] and row['forced_ids']==[],'natural output contains forced context')
    ids=checked_ids(row['action_ids'],row['stop_reason']);require(row['remaining_budget']==3084,'natural generation cap changed')
    text=tokenizer.decode(ids,skip_special_tokens=False);require(text==row['text'],'native token/text mismatch')
    parsed=native_record(text,frozen['case'],frozen['baseline'],row['stop_reason'])
    require(parsed==row['parsed'],'cold native parser mismatch')
    return parsed


def owner_change(after,before):
    return dict(gained=sorted(set(after)-set(before)),lost=sorted(set(before)-set(after)),retained=sorted(set(after)&set(before)))


def consume(path,packet,tokenizer,expected_ids):
    records=rows(path);seen=set();frozen=indexed(packet['records'],'example_id')
    for row in records:
        eid=row['example_id'];require(eid not in seen and eid in frozen,'duplicate/unknown natural output');seen.add(eid)
        require(row['manifest_sha256']==digest(packet),'natural manifest identity')
        old=frozen[eid];parsed=validate_natural(row,old,tokenizer)
        row['score']=score(parsed,seed=-1,length=len(row['action_ids']),stop=row['stop_reason'])
        row['owner_changes']={t:owner_change(row['score'][t]['owners'],old['baseline_score'][t]['owners']) for t in ('50','60','80')}
        row['direct_incidence']={t:incidence(parsed,threshold=int(t)/100) for t in ('50','60','80')}
        row['matching_reassignment']={t:dict(
            gained=[o for o in row['owner_changes'][t]['gained'] if incidence(old['baseline'],threshold=int(t)/100)[o]],
            lost=[o for o in row['owner_changes'][t]['lost'] if row['direct_incidence'][t][o]]) for t in ('50','60','80')}
        if old['split']=='train':
            case=old['bridge_case'];ent=old['entrance'];ids=row['action_ids'];index=ent['action_index'];reached=ids[:index]==ent['state_ids']
            source_prefix=native_record(tokenizer.decode(ent['state_ids'],skip_special_tokens=False),old['case'],old['baseline'],'conditional')
            actual_prefix=native_record(tokenizer.decode(ids[:index],skip_special_tokens=False),old['case'],old['baseline'],'conditional')
            prefix_scores=[score(p,seed=-1,length=index,stop='conditional') for p in (source_prefix,actual_prefix)]
            row['training_readout']=dict(owner=case['owner'],target_global={t:case['owner'] in row['score'][t]['owners'] for t in ('50','60','80')},
                target_direct={t:bool(row['direct_incidence'][t][case['owner']]) for t in ('50','60','80')},
                exact_entrance_reached=reached,A_selected_at_exact_entrance=reached and len(ids)>index and ids[index]==ent['A_id'],
                observed_next_token_at_exact_entrance=ids[index] if reached and len(ids)>index else None,
                first_divergence_from_Source=first_fork(ids,old['baseline_ids']),
                earlier_token_differences=[dict(action_index=i,source_id=a,candidate_id=b) for i,(a,b) in enumerate(zip(ent['state_ids'],ids[:index])) if a!=b],
                candidate_prefix_action_tokens=min(len(ids),index),source_prefix_owner_ids=prefix_scores[0]['50']['owners'],
                candidate_prefix_owner_ids=prefix_scores[1]['50']['owners'],prefix_owner_changes=owner_change(prefix_scores[1]['50']['owners'],prefix_scores[0]['50']['owners']),
                prefix_parser_drops=dict(source=source_prefix['dropped_prediction_count'],candidate=actual_prefix['dropped_prediction_count']),
                prefix_comparison_scope='Same action-token horizon, possibly partial row; exact-token mismatch is not semantic-state absence.')
    require(seen==set(expected_ids),'missing natural outputs')
    return records


def aggregate_scores(scores):
    out={}
    for t in ('50','60','80'):
        tp,fp,fn=[sum(s[t][k] for s in scores) for k in ('tp','fp','fn')]
        out[t]=dict(tp=tp,fp=fp,fn=fn,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)
    out.update({k:sum(s[k] for s in scores) for k in ('prediction_count','parsed_prediction_count','invalid_predictions','parser_drops','complete_token_length','strict_repeats','cap')})
    return out


def reduce_records(records,packet):
    result={}
    for split,count in [('train',2),('guard',16)]:
        frozen=[r for r in packet['records'] if r['split']==split];ids={r['example_id'] for r in frozen}
        selected=[r for r in records if r['example_id'] in ids];require(len(selected)==count,'split output coverage')
        result[split]=dict(images=count,source=aggregate_scores([r['baseline_score'] for r in frozen]),candidate=aggregate_scores([r['score'] for r in selected]),
            owner_changes={t:{kind:sum(len(r['owner_changes'][t][kind]) for r in selected) for kind in ('gained','lost','retained')} for t in ('50','60','80')},
            eos=sum(r['stop_reason']=='im_end' for r in selected),case_ids=sorted(ids))
        if split=='train':result[split]['targets']={r['example_id']:r['training_readout'] for r in selected}
    return dict(schema='native_entrance_ce_eval_reduction.v1',splits=result,
        limitation='Two trained images plus16 historically used development guards; not untouched transfer or a promoted checkpoint.')


def execute(output,receipt_path):
    import numpy as np
    import torch
    from src.config.inference import load_research_infer_config
    from src.qwen.native import prepare_native_inputs,prepare_replay
    from src.qwen.generation import generate_continuations
    from .runtime import load_policy
    packet=json.loads((output/'manifest.json').read_text());training=json.loads(receipt_path.read_text())
    adapter=validate_receipt(training,packet)
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='1' and torch.cuda.device_count()==1,'GPU1 only')
    run=output/'execution';run.mkdir(exist_ok=False)
    started=time.monotonic();terminal=dict(status='running',pid=os.getpid(),model_loads=0,score_forwards=0,model_forwards=0,image_forwards=0,
        continuations=0,new_tokens=0,manifest_sha256=digest(packet),training_receipt_path=str(receipt_path),training_receipt_sha256=file_hash(receipt_path))
    publish(run/'launch.json',terminal)
    def expired(*_):raise TimeoutError('1800 second cumulative evaluator budget')
    signal.signal(signal.SIGALRM,expired);signal.alarm(1800)
    try:
        for p,sha in packet['source_files'].items():require(file_hash(p)==sha,'frozen evaluation source changed')
        config=checkpoint_config(load_research_infer_config(CONFIG).config,adapter['root'])
        require(str(config.model.base_model)==packet['source_model']['base_model_path'] and str(config.embedding_delta.path)==packet['source_model']['source_embedding']['root'],'candidate base/embedding config')
        qwen,identity=load_policy(config,device=torch.device('cuda:0'));terminal['model_loads']=1
        require(identity['model_identity']['adapter']['adapter_path']==adapter['root'] and not identity['model_identity']['adapter']['merged_adapters'],'saved unmerged candidate identity')
        require(identity['effective_settings']['observed_model_dtype']['parameter_dtype_names']==['torch.float32'] and identity['effective_settings']['observed_attn_implementation']=='sdpa','observed FP32 SDPA')
        publish(run/'model.json',identity);publish(run/'config.json',config.model_dump(mode='json'))
        publish(run/'code.json',{str(p):file_hash(p) for p in [Path(__file__),Path(__file__).with_name('branch_bridge.py'),Path(__file__).with_name('route_access.py'),CONFIG]})
        qwen.model.eval();torch.cuda.reset_peak_memory_stats()
        def count_model(*_):terminal['model_forwards']+=1
        def count_image(*_):terminal['image_forwards']+=1
        qwen.model.register_forward_pre_hook(count_model);visual=[m for n,m in qwen.model.named_modules() if n.endswith('visual')]
        require(len(visual)==1,'visual counter identity');visual[0].register_forward_pre_hook(count_image)
        requests={}
        for frozen in packet['records']:
            request,_=build_requests(qwen,packet['configs'][frozen['split']],[frozen['case']])
            require(list(request[0].expected_token_ids)==frozen['prompt_token_ids'],'cold reconstructed prompt changed')
            requests[frozen['example_id']]=request[0]
        def prepare_batch(frozen):
            batch=prepare_native_inputs(qwen.processor,[requests[frozen['example_id']]],device=torch.device('cuda:0'),record_media_identity=True)
            require(list(batch.prompt_token_ids[0])==frozen['prompt_token_ids'] and batch.media_sha256[0]==frozen['case']['image_plan']['executed_media_sha256'],'cold prompt/media identity')
            return batch
        # Complete both checkpoint-score checks before ANY natural outcome.
        for frozen in [r for r in packet['records'] if r['split']=='train']:
            batch=prepare_batch(frozen);ent=frozen['entrance'];cid=frozen['bridge_case']['case_id']
            with torch.inference_mode():
                replay=prepare_replay(qwen.model,batch.inputs,prompt_token_ids=frozen['prompt_token_ids'],continuation_token_ids=ent['state_ids']+[ent['A_id']])
                terminal['score_forwards']+=1
                logits=replay.aligned_logits(qwen.model(**replay.inputs).logits)
                require(logits.shape[0]==ent['action_index']+1 and int(replay.target_ids[-1])==ent['A_id'],'cold score causal alignment')
                values=logits[-1].detach().float().cpu().numpy().copy();del logits,replay
            path=run/f"{cid.replace(':','_')}-cold-entry-logits.npy"
            with path.open('xb') as stream:np.save(stream,values,allow_pickle=False)
            observed=summarize_logits(np.load(path,allow_pickle=False),ent);saved=training['final_scores'][cid]
            deltas={k:observed[k]-saved[k] for k in ('logprob','A_vs_best_other_margin')}
            require(all(abs(v)<=1e-5 for v in deltas.values()) and observed['A_id']==saved['target_id'] and
                all(observed[k]==saved[k] for k in ('rank_min','top1_id')),'saved checkpoint final score mismatch')
            publish(run/f"{cid.replace(':','_')}-cold-entry.json",dict(readout=observed,trainer=saved,deltas=deltas,tolerance=1e-5,logits_sha256=file_hash(path)))
        require(terminal['score_forwards']==2,'cold fixed-state score coverage')
        path=run/'rows.jsonl';expected=[]
        with path.open('x') as stream:
            for frozen in packet['records']:
                batch=prepare_batch(frozen);tick=time.monotonic()
                require(terminal['continuations']<18 and terminal['new_tokens']+3084<=55512,'natural execution bound')
                result=generate_continuations(qwen.model,batch,extensions=[[]],budgets=[3084],eos_token_id=151645,pad_token_id=qwen.tokenizer.pad_token_id,trace='none')[0]
                require(result.request_id==frozen['example_id'],'natural request association');terminal['continuations']+=1;terminal['new_tokens']+=len(result.token_ids)
                ids=list(result.token_ids);text=qwen.tokenizer.decode(ids,skip_special_tokens=False)
                row=dict(example_id=frozen['example_id'],split=frozen['split'],manifest_sha256=digest(packet),action_ids=ids,prefix_ids=[],forced_ids=[],remaining_budget=3084,
                    text=text,stop_reason=result.stop_reason,seconds=time.monotonic()-tick,
                    parsed=native_record(text,frozen['case'],frozen['baseline'],result.stop_reason))
                validate_natural(row,frozen,qwen.tokenizer);stream.write(json.dumps(row)+'\n');stream.flush();os.fsync(stream.fileno());expected.append(row['example_id'])
        cold=consume(path,packet,qwen.tokenizer,expected);publish(run/'consumer.json',cold);publish(run/'reduction.json',reduce_records(cold,packet))
        require(terminal['continuations']==18,'natural output count');terminal['status']='completed'
    except BaseException as exc:
        terminal.update(status='failed',error=repr(exc),traceback=traceback.format_exc());raise
    finally:
        signal.alarm(0);terminal.update(elapsed_seconds=time.monotonic()-started,rss_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0,
            artifact_bytes=sum(p.stat().st_size for p in output.rglob('*') if p.is_file()))
        publish(run/'terminal.json',terminal)


def verify(output):
    import numpy as np
    from tokenizers import Tokenizer
    packet=json.loads((output/'manifest.json').read_text());run=output/'execution';terminal=json.loads((run/'terminal.json').read_text())
    require(terminal['status']=='completed' and terminal['model_loads']==1 and terminal['score_forwards']==2 and terminal['continuations']==18
        and terminal['new_tokens']<=55512 and terminal['elapsed_seconds']<1800,'evaluation terminal bounds')
    training=json.loads(Path(terminal['training_receipt_path']).read_text());validate_receipt(training,packet)
    require(file_hash(terminal['training_receipt_path'])==terminal['training_receipt_sha256'],'sealed receipt changed')
    for p,sha in packet['source_files'].items():require(file_hash(p)==sha,'frozen evaluation source changed')
    tokenizer=Tokenizer.from_file(packet['source_model']['base_model_path']+'/tokenizer.json')
    cold=consume(run/'rows.jsonl',packet,tokenizer,[r['example_id'] for r in packet['records']])
    require(cold==json.loads((run/'consumer.json').read_text()),'CPU native consumer differs')
    require(reduce_records(cold,packet)==json.loads((run/'reduction.json').read_text()),'CPU split reduction differs')
    require(sum(len(r['action_ids']) for r in cold)==terminal['new_tokens'],'generated token counter differs')
    for r in packet['records']:
        if r['split']!='train':continue
        cid=r['bridge_case']['case_id'];stem=cid.replace(':','_');saved=json.loads((run/f'{stem}-cold-entry.json').read_text())
        values=np.load(run/f'{stem}-cold-entry-logits.npy',allow_pickle=False)
        require(summarize_logits(values,r['entrance'])==saved['readout'],'saved full-vocab cold score differs')
        require(saved['trainer']==training['final_scores'][cid] and all(abs(v)<=1e-5 for v in saved['deltas'].values()),'training/cold score evidence')
    paths=[Path(__file__),output/'manifest.json',run/'rows.jsonl',run/'consumer.json',run/'reduction.json']
    return dict(status='candidate_cpu_verified',natural_outputs=18,fixed_state_score_checks=2,consumer_equal=True,reduction_equal=True,
        files_sha256={str(p):file_hash(p) for p in paths})


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','execute','verify']);p.add_argument('--output',type=Path,default=OUTPUT)
    p.add_argument('--resume-selection',action='store_true')
    p.add_argument('--receipt',type=Path,default=ROOT_OUTPUT/'training/receipt.json')
    args=p.parse_args()
    if args.command=='prepare':prepare(args.output,resume_selection=args.resume_selection)
    elif args.command=='execute':execute(args.output,args.receipt)
    else:print(json.dumps(verify(args.output),indent=2))


if __name__=='__main__':main()
