"""Eight frozen single-token Source releases; conditional supply, never training admission."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import resource
import signal
import time

from src.artifacts import load_canonical_json
from src.data.geometry import iou_xyxy
from probes.dora_owner_learning.candidate_opportunity import digest,file_hash,indexed,require,rows,score,validate_parser
from probes.dora_owner_learning.route_access import ROOT,SOURCE_ROOT,CONFIG,checked_ids,publish,first_fork
from probes.dora_owner_learning.round1_realization import trace_card
from probes.dora_owner_learning.reward_rows import _gt_objects,_pred_objects
from probes.source_rweak_row_cross.run import native_record

OUT=ROOT/'2026-09-10-selective-owner-learning-autonomous/positive-branch-expansion'
INVENTORY=OUT.parent/'positive-support-inventory/inventory.json'
INVENTORY_SHA='7bddf95ea074085b951ca478584be4323e01a7305d8ea51e2a5c37b29ac7d817'
FROZEN={'252411':('1676022',{2026090602:(1,50629)}),'465695':('1612233',{2026090601:(1,2721)}),
        '496747':('1614837',{2026090602:(7,152280),2026090603:(7,152283),2026090604:(7,152301)}),
        '529411':('2147505',{2026090603:(1,6565)}),'538814':('1121312',{2026090602:(4,151933)}),
        '540567':('1491504',{2026090601:(1,65)})}
EOS,PAD,START,END,CAP=151645,151643,151646,151649,3084


def freeze_boundary(variant,source_ids,sample_ids):
    index,target=variant['action_index'],variant['target_token_id']
    prefix=variant['source_action_prefix_ids']
    require(type(index) is int and 0<=index<min(len(source_ids),len(sample_ids)) and len(prefix)==index,
            'shifted native entry index')
    require(prefix==source_ids[:index]==sample_ids[:index] and digest(prefix)==variant['source_action_prefix_sha256'],
            'native/sample prefix mismatch')
    require(first_fork(source_ids,sample_ids)==index and source_ids[index]==variant['source_token_id'] and
            sample_ids[index]==target and source_ids[index]!=target,'first-fork/target mismatch')
    extension=prefix+[target]
    require(EOS not in extension and PAD not in extension and extension[0]==START,'nonterminal row prefix required')
    return dict(prefix_ids=prefix,forced_ids=[target],extension_ids=extension,entry_index=index,
                free_start=len(extension),remaining_budget=CAP-len(extension))


def first_row_evidence(ids,entry_index,parsed,card,case,tokenizer):
    starts=[i for i,t in enumerate(ids[:entry_index+1]) if t==START]
    require(starts,'supplied token is not inside a row')
    start=starts[-1]
    next_start=next((i for i in range(entry_index+1,len(ids)) if ids[i] in (START,EOS)),len(ids))
    close=next((i for i in range(entry_index+1,next_start) if ids[i]==END),None)
    end=close+1 if close is not None else next_start
    text=tokenizer.decode(ids[start:end],skip_special_tokens=False)
    isolated=native_record(text,{'row_id':case['example_id']},case['golden'],'conditional')
    row_valid=close is not None and isolated['dropped_prediction_count']==0 and len(isolated['pred'])==1
    char_start=len(tokenizer.decode(ids[:start],skip_special_tokens=False));char_end=len(tokenizer.decode(ids[:end],skip_special_tokens=False))
    parsed_indices=[i for i,p in enumerate(parsed['pred']) if p['char_start']==char_start and p['char_end']==char_end and p['raw_span_text']==text]
    if row_valid:require(len(parsed_indices)==1,'actual first row cannot be aligned to full parser')
    projection=[i for i,obj in enumerate(parsed['pred']) if _pred_objects(dict(parsed,pred=[obj]))[0]]
    assigned={}
    for threshold in ('50','60','80'):
        assigned[threshold]=any(m['owner']==case['owner'] and projection[m['pred_index']] in parsed_indices for m in card[threshold]['matches']) if row_valid else False
    gt=_gt_objects(case['golden'],row_id=case['example_id']);gt_index=next(i for i,g in enumerate(case['golden']['gt']) if str(g['object_id'])==case['owner'])
    pred,_=_pred_objects(isolated)
    direct_iou=iou_xyxy(pred[0][1],gt[gt_index][1]) if row_valid and pred and pred[0][0]==gt[gt_index][0] else 0.
    return dict(start_token=start,end_token_exclusive=end,complete=close is not None,valid=row_valid,
                row_token_ids=ids[start:end],row_text=text,parsed_prediction_indices=parsed_indices,
                full_global_target_assignment=assigned,direct_target_iou=direct_iou,
                isolated_parsed=isolated,free_tokens_in_row=ids[entry_index+1:end],free_suffix_after_row_ids=ids[end:])


def eligibility(source,candidate,first_row,stop):
    reasons=[]
    if not first_row['valid'] or not first_row['full_global_target_assignment']['50']:
        reasons.append('actual_entry_row_not_globally_assigned_to_target_at50')
    if not set(source['50']['owners'])<=set(candidate['50']['owners']):reasons.append('lost_Source_IoU50_owner')
    if candidate['50']['fp']>source['50']['fp']:reasons.append('annotation_relative_FP_increased')
    if candidate['strict_repeats']>source['strict_repeats']:reasons.append('strict_repeats_increased')
    if candidate['parser_drops']>source['parser_drops']:reasons.append('parser_drops_increased')
    if stop!='im_end':reasons.append('not_natural_EOS')
    return dict(candidate_eligible=not reasons,rejection_reasons=reasons,
                authority='Conditional prospective eligibility only; lead training-label admission still required.')


def reduce_record(record,variant,tokenizer):
    boundary=variant['boundary'];ids=record['action_ids']
    require(record['variant_id']==variant['variant_id'] and record['prefix_ids']==boundary['prefix_ids'] and
            record['forced_ids']==boundary['forced_ids'] and ids==boundary['extension_ids']+record['free_ids'] and
            record['free_start']==boundary['free_start'] and record['remaining_budget']==boundary['remaining_budget'], 'forced/free boundary identity')
    checked_ids(ids,record['stop_reason'])
    require(len(record['free_ids'])<=boundary['remaining_budget'] and tokenizer.decode(ids,skip_special_tokens=False)==record['text'],'complete action/text/budget identity')
    c=variant['case'];parsed=native_record(record['text'],{'row_id':c['example_id']},c['golden'],record['stop_reason'])
    require(parsed==record['parsed'],'cold native parser identity')
    candidate=score(parsed,seed=variant['seed'],length=len(ids),stop=record['stop_reason']);source=variant['source_score']
    first=first_row_evidence(ids,boundary['entry_index'],parsed,candidate,c,tokenizer)
    changes={}
    for t in ('50','60','80'):
        old,new=set(source[t]['owners']),set(candidate[t]['owners'])
        changes[t]=dict(gained=sorted(new-old),lost=sorted(old-new),retained=sorted(new&old),
                        source_owners=sorted(old),candidate_owners=sorted(new),
                        delta={k:candidate[t][k]-source[t][k] for k in ('tp','fp','fn','f1')})
    return dict(variant_id=variant['variant_id'],case_id=c['case_id'],image_id=c['image_id'],owner=c['owner'],seed=variant['seed'],
                source=source,candidate=candidate,actual_first_row=first,owner_changes=changes,
                eligibility=eligibility(source,candidate,first,record['stop_reason']),
                conditional=True,forced_context_length=boundary['entry_index'],forced_token_count=1,free_token_count=len(record['free_ids']))


def choose_lowest(results):
    passing=defaultdict(list)
    for r in results:
        if r['eligibility']['candidate_eligible']:passing[r['case_id']].append(r)
    return {cid:min(values,key=lambda r:r['seed'])['variant_id'] for cid,values in passing.items()}


def prepare(output=OUT):
    from tokenizers import Tokenizer
    require(not (output/'inputs.json').exists(),'occupied input packet')
    require(file_hash(INVENTORY)==INVENTORY_SHA,'frozen inventory bytes changed')
    inventory=json.loads(INVENTORY.read_text());sources={str(INVENTORY):INVENTORY_SHA}
    for path,sha in inventory['input_files'].items():require(file_hash(path)==sha,'inventory source changed');sources[path]=sha
    route_path=ROOT/'2026-09-10-fixed-witness-route-access/inputs.json';route=load_canonical_json(route_path)
    groups=indexed(route['groups'],'example_id');routes=indexed(route['routes'],'route_id');witnesses=indexed(route['witnesses'],'witness_id')
    manifest=json.loads((SOURCE_ROOT/'run_manifest.json').read_text());raw=indexed(rows(SOURCE_ROOT/'gt_vs_pred.jsonl'),'row_id')
    images=indexed(rows(SOURCE_ROOT/'image_plan.jsonl'),'row_id');prompts=indexed(manifest['prompt_trace'],'row_id')
    for path in [SOURCE_ROOT/'run_manifest.json',SOURCE_ROOT/'pred_token_trace.jsonl',SOURCE_ROOT/'image_plan.jsonl',CONFIG,Path(__file__),Path(__file__).with_name('test_runner.py')]:sources[str(path)]=file_hash(path)
    tokenizer_path=Path(route['plan']['model']['base_model_path'])/'tokenizer.json'
    require(file_hash(tokenizer_path)==manifest['frontend_identity']['tokenizer_sha256'],'tokenizer identity')
    tok=Tokenizer.from_file(str(tokenizer_path));sources[str(tokenizer_path)]=file_hash(tokenizer_path)
    selected=indexed(inventory['shortlisted_owners'],'image_id');require(set(selected)==set(FROZEN),'six-owner population')
    traces=defaultdict(list)
    for t in rows(SOURCE_ROOT/'pred_token_trace.jsonl'):
        if t['trace_type']=='generated_token' and t['row_id'] in {c['example_id'] for c in selected.values()}:traces[t['row_id']].append(t)
    shards={};variants=[]
    for image_id,(owner,seeds) in FROZEN.items():
        c=selected[image_id];require(c['owner']==owner,'frozen owner');g=groups[c['example_id']];gold=raw[c['example_id']]
        require(next(x for x in gold['gt'] if str(x['object_id'])==owner)==c['gt_object'],'immutable target GT')
        require(file_hash(c['image_path'])==c['image_sha256']==g['image_content_sha256']==images[c['example_id']]['image_content_sha256'] and
                c['executed_media_sha256']==g['executed_media_sha256']==images[c['example_id']]['executed_media_sha256'] and
                c['prompt_token_ids_sha256']==digest(g['prompt_token_ids'])==prompts[c['example_id']]['backend_executed_prompt_token_ids_sha256'], 'Source prompt/media/image identity')
        source_card=trace_card(gold,traces[c['example_id']],tok);source_ids=[t['token_id'] for t in sorted(traces[c['example_id']],key=lambda t:t['generated_step_index']) if not t['is_pad']]
        vv={v['seed']:v for v in c['variants']};require(set(vv)==set(seeds),'eight frozen seed variants')
        for seed,(index,target) in seeds.items():
            v=vv[seed];require(v['action_index']==index and v['target_token_id']==target and tok.decode([target],skip_special_tokens=False)==v['target_token'],'frozen token/index')
            sample=routes[v['route_id']]['ids'];require(routes[v['source_greedy_route_id']]['ids']==source_ids and digest(sample)==v['sample_action_ids_sha256'],'retained route identity')
            witness=witnesses[v['witness_id']];require(witness['route_id']==v['route_id'] and witness['seed']==seed,'witness selector')
            path=Path(v['raw_candidate_path'])
            require(file_hash(path)==v['raw_candidate_sha256'],'raw sample shard identity');sources[str(path)]=v['raw_candidate_sha256']
            if str(path) not in shards:shards[str(path)]=json.loads(path.read_text())
            matches=[x for x in shards[str(path)]['rollouts'] if x['example_id']==c['example_id'] and x['seed']==seed]
            require(len(matches)==1,'raw sample cell uniqueness');sample_raw=matches[0]
            require(sample_raw['generated_token_ids']+[EOS]==sample and sample_raw['stop_reason']=='im_end' and
                    sample_raw['prompt_token_ids']==g['prompt_token_ids'] and sample_raw['executed_media_sha256']==g['executed_media_sha256'] and
                    digest(sample_raw['predictions'])==v['candidate_parser_sha256'],'sample tokens/prompt/parser identity')
            validate_parser(sample_raw['generated_text'],sample_raw['predictions'],gold['image_width'],gold['image_height'])
            require(tok.decode(sample[:-1],skip_special_tokens=False)==sample_raw['generated_text'] and
                    sample[:len(v['complete_row_token_ids'])]==v['complete_row_token_ids'] and
                    digest(v['complete_row_token_ids'])==v['complete_row_token_sha256'] and
                    tok.decode(v['complete_row_token_ids'],skip_special_tokens=False)==v['complete_row_text'],'complete row witness identity')
            boundary=freeze_boundary(v,source_ids,sample)
            variants.append(dict(variant_id=f'{image_id}:{owner}:{seed}',seed=seed,boundary=boundary,source_ids=source_ids,
                source_score=source_card,case=dict(case_id=c['case_id'],example_id=c['example_id'],image_id=image_id,owner=owner,category=c['category'],golden=gold,group=g),
                sampled_witness=v,sampled_action_ids=sample))
    require(len(variants)==8 and sum(v['boundary']['remaining_budget'] for v in variants)==24635,'frozen continuation budget')
    require(inventory['source_checkpoint']['adapter_fingerprint']==route['plan']['model']['current_adapter']['fingerprint'],'Source checkpoint fingerprint')
    output.mkdir(parents=True,exist_ok=True)
    packet=dict(schema='positive_branch_expansion.inputs.v1',inventory_sha256=INVENTORY_SHA,variants=variants,
        model=route['plan']['model'],train_source=route['plan']['sources']['train_jsonl'],source_files=sources,
        numerical_policy=dict(dtype='fp32',attention='sdpa',patch_linearization='enabled',temperature=0.,top_p=1.,repetition_penalty=1.,total_action_cap=3084),
        limits=dict(model_loads=1,continuations=8,image_forwards=8,max_new_tokens=24635,max_model_forwards=24643,seconds=3600),
        qualifications='Lead viewed all6 Source images. Small visible hairdrier/bottle; tiger-shaped plush retains teddy bear annotation; luggage-occluded near-full-frame bed; table/oven may be localization repair. No uniform missing-instance claim.',
        preservation48_overlap=['529411'],overlap_ruling='No current support/mask changes; lead resolves only if prospective label passes and a new training packet is frozen.',
        admission_boundary='Conditional evidence only, no training labels admitted by this producer.')
    publish(output/'inputs.json',packet)
    return packet


def reduce_all(output=OUT):
    from tokenizers import Tokenizer
    p=load_canonical_json(output/'inputs.json');tok=Tokenizer.from_file(str(Path(p['model']['base_model_path'])/'tokenizer.json'))
    paths=list((output/'rows').glob('*.json'));require(len(paths)==8,'complete eight-output coverage')
    records=indexed([load_canonical_json(path) for path in paths],'variant_id')
    require(set(records)=={v['variant_id'] for v in p['variants']},'exact variant output identities')
    results=[reduce_record(records[v['variant_id']],v,tok) for v in p['variants']]
    return dict(schema='positive_branch_expansion.reduction.v1',conditional_variants=8,owners=6,results=results,
                passing_variants=sum(r['eligibility']['candidate_eligible'] for r in results),prospective_selection_by_lowest_seed=choose_lowest(results),
                selected_owners=len(choose_lowest(results)),preservation48_overlap=p['preservation48_overlap'],
                status='prospective_candidates_require_lead_admission',no_training=True,no_autonomous_output_claim=True)


def execute(output=OUT):
    import torch
    from src.config.inference import load_research_infer_config
    from src.data import load_raw_examples
    from src.qwen.generation import generate_continuations,NativeGenerationPolicy
    from probes.dora_owner_learning.runtime import load_policy,build_request,materialize
    p=load_canonical_json(output/'inputs.json');require(os.environ.get('CUDA_VISIBLE_DEVICES')=='0' and not (output/'launch.json').exists(),'single GPU0 invocation')
    for path,sha in p['source_files'].items():require(file_hash(path)==sha,f'frozen source changed: {path}')
    cfg=load_research_infer_config(CONFIG).config
    require(str(cfg.model.base_model)==p['model']['base_model_path'] and str(cfg.adapter.path)==p['model']['current_adapter']['root'] and
            str(cfg.embedding_delta.path)==p['model']['source_embedding']['root'],'original Source only')
    require(cfg.model.dtype=='fp32' and cfg.backend.hf.attn_implementation=='sdpa' and cfg.backend.hf.patch_embed_linearization=='enabled','frozen numerics')
    require(file_hash(cfg.data.input_jsonl)==p['train_source']['sha256'],'original train input')
    for identity in (p['model']['current_adapter'],p['model']['source_embedding']):
        for f in identity['files']:require(file_hash(Path(identity['root'])/f['relative_path'])==f['sha256'],'Source model payload')
    publish(output/'launch.json',dict(pid=os.getpid(),started=time.time(),visible_device='0',inputs_sha256=file_hash(output/'inputs.json')))
    started=time.monotonic();counts=dict(model_loads=0,continuations=0,image_forwards=0,model_forwards=0,new_tokens=0);status,error='failed',None
    def expired(*_):raise TimeoutError('3600-second conditional invocation bound')
    signal.signal(signal.SIGALRM,expired);signal.alarm(3600)
    try:
        qwen,identity=load_policy(cfg,device=torch.device('cuda:0'));counts['model_loads']=1;qwen.model.eval()
        require(identity['effective_settings']['observed_model_dtype']['parameter_dtype_names']==['torch.float32'] and identity['effective_settings']['observed_attn_implementation']=='sdpa' and
                not identity['model_identity']['adapter']['merged_adapters'],'observed Source execution')
        publish(output/'loaded_model.json',identity);publish(output/'effective_config.json',cfg.model_dump(mode='json'))
        for parameter in qwen.model.parameters():parameter.requires_grad_(False)
        def model_count(*_):counts['model_forwards']+=1;require(counts['model_forwards']<=24643,'model forward budget')
        def image_count(*_):counts['image_forwards']+=1;require(counts['image_forwards']<=8,'image forward budget')
        qwen.model.register_forward_pre_hook(model_count)
        visual=[m for n,m in qwen.model.named_modules() if n.endswith('visual')];require(len(visual)==1,'visual counter identity');visual[0].register_forward_pre_hook(image_count)
        raw={str(r.example_id):r for r in load_raw_examples(cfg.data.input_jsonl)};torch.cuda.reset_peak_memory_stats()
        policy=NativeGenerationPolicy(temperature=0.,top_p=1.,repetition_penalty=1.,top_k=0,use_model_defaults=False)
        for v in p['variants']:
            c,g,b=v['case'],v['case']['group'],v['boundary'];request,im,prompt=build_request(raw[c['example_id']],config=cfg,qwen=qwen,row_index=c['golden']['row_index'])
            batch=materialize(qwen,request)
            require(list(batch.prompt_token_ids[0])==g['prompt_token_ids'] and batch.media_sha256[0]==g['executed_media_sha256'] and
                    list(batch.image_grids[0])==g['observed_image_grid_thw'] and im.image_content_sha256==g['image_content_sha256'],'live Source prompt/media/grid')
            with torch.inference_mode():
                result=generate_continuations(qwen.model,batch,extensions=[b['extension_ids']],budgets=[b['remaining_budget']],
                    eos_token_id=EOS,pad_token_id=qwen.tokenizer.pad_token_id,policy=policy,trace='none')[0]
            require(result.request_id==c['example_id'],'continuation identity');torch.cuda.synchronize()
            counts['continuations']+=1;counts['new_tokens']+=len(result.token_ids);require(counts['new_tokens']<=24635,'new token budget')
            ids=b['extension_ids']+list(result.token_ids);text=qwen.tokenizer.decode(ids,skip_special_tokens=False)
            record=dict(variant_id=v['variant_id'],prefix_ids=b['prefix_ids'],forced_ids=b['forced_ids'],free_start=b['free_start'],
                        free_ids=list(result.token_ids),remaining_budget=b['remaining_budget'],action_ids=ids,text=text,stop_reason=result.stop_reason,
                        parsed=native_record(text,{'row_id':c['example_id']},c['golden'],result.stop_reason))
            path=output/'rows'/f"{c['image_id']}-{v['seed']}.json";path.parent.mkdir(exist_ok=True);publish(path,record)
            reduced=reduce_record(load_canonical_json(path),v,qwen.tokenizer)
            publish(output/'rows'/f"{c['image_id']}-{v['seed']}.reduction",reduced)
            print(json.dumps(dict(variant=v['variant_id'],new_tokens=len(result.token_ids),eligibility=reduced['eligibility'],seconds=time.monotonic()-started)),flush=True)
        require(counts['continuations']==counts['image_forwards']==8,'eight complete continuations/images')
        reduction=reduce_all(output);publish(output/'reduction.json',reduction)
        receipt=dict(schema='positive_branch_expansion.receipt.v1',status='completed_for_lead_admission',inputs_sha256=file_hash(output/'inputs.json'),
                     reduction_sha256=file_hash(output/'reduction.json'),counts=counts,model_seconds=time.monotonic()-started,
                     peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
                     peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,training_steps=0,
                     prospective_passing_variants=reduction['passing_variants'],prospective_selected_owners=reduction['selected_owners'],
                     preservation48_overlap=p['preservation48_overlap'],labels_admitted_by_producer=False)
        publish(output/'receipt.json',receipt);status='completed'
    except BaseException as exc:error=f'{type(exc).__name__}: {exc}';raise
    finally:
        signal.alarm(0);publish(output/'terminal.json',dict(status=status,error=error,counts=counts,model_seconds=time.monotonic()-started))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','execute','verify']);a=parser.parse_args()
    if a.command=='prepare':p=prepare();print(json.dumps(dict(variants=len(p['variants']),limits=p['limits'])))
    elif a.command=='execute':execute()
    else:
        r=reduce_all();require(r==load_canonical_json(OUT/'reduction.json'),'CPU reduction replay differs')
        require(load_canonical_json(OUT/'terminal.json')['status']=='completed','incomplete run')
        print(json.dumps(dict(variants=r['conditional_variants'],passing_variants=r['passing_variants'],selected=r['prospective_selection_by_lowest_seed'])))
