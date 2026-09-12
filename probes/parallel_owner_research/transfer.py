"""Fixed fresh256 transfer, explicit checkpoint labels and immutable native helpers."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import sys
import time
import traceback

from probes.dora_owner_learning import margin_preserved_endpoint as accepted
from probes.dora_owner_learning.candidate_opportunity import file_hash, require, score
from probes.dora_owner_learning.entrance_ce_eval import aggregate_scores
from probes.dora_owner_learning.geometric_dedup_eval import overlap_counts

WORKTREE = Path('/data/CoordExp/.worktrees/research-probes')
INVESTIGATION = Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration')
ROOT = INVESTIGATION / '2026-09-12-parallel-owner-research/transfer'
PRIOR = INVESTIGATION / '2026-09-11-positive-progress-matched-control'
SOURCE = Path('/data/CoordExp/public_data/coco/rescale_32_1024_bbox_len12000/val.coord.jsonl')
NATIVE_SOURCE = Path('/data/CoordExp/public_data/coco/rescale_32_1024_bbox_len12000_xy_sorted/val.coord.jsonl')
SALT = 'parallel-owner-fresh256-2026-09-12:'
LABELS = ('Stable50', 'C32', 'D17')
CAP = 3084
read = accepted.load_json
publish = accepted.publish
digest = accepted.digest


def image_ids(value, key=''):
    """Conservative identity-only projection; no model metric/output selection."""
    result = set()
    if isinstance(value, dict):
        for k, v in value.items():
            result.update(image_ids(v, k))
    elif isinstance(value, list):
        for v in value:
            result.update(image_ids(v, key))
    elif type(value) is int and (key == 'image_id' or key.endswith('image_ids')):
        result.add(value)
    elif isinstance(value, str):
        result.update(int(m.group(1)) for m in re.finditer(
            r'(?:coco2017_(?:train|val)_|(?:train|val)2017/|image[-_]?)(\d{1,12})(?=[^\d]|$)', value))
    return result


def select_ids(universe, exclusions, count=256):
    eligible = set(universe) - set(exclusions)
    require(len(eligible) >= count, 'insufficient unseen panel; no alternate source')
    return sorted(eligible, key=lambda i: hashlib.sha256(f'{SALT}{i}'.encode()).hexdigest())[:count]


def freeze_selection():
    require(not (ROOT / 'selection.json').exists(), 'selection already frozen')
    paths = [Path(p) for p in subprocess.check_output(
        ['rg', '--files', '--no-ignore', str(INVESTIGATION)], text=True).splitlines()]
    markers = ('manifest', 'selection', 'cases', 'inputs', 'train.', 'dev.',
               'holdout', 'confirmation', 'execution-requests')
    paths = sorted(p for p in paths if p.suffix in ('.json', '.jsonl')
                   and '2026-09-12-parallel-owner-research' not in str(p)
                   and any(m in p.name for m in markers))
    # Include input JSONLs explicitly named by old manifests, not full source corpora.
    excluded, sources = set(), []
    for path in paths:
        values = ([json.loads(line) for line in path.open() if line.strip()]
                  if path.suffix == '.jsonl' else read(path))
        found = image_ids(values)
        excluded.update(found)
        sources.append(dict(path=str(path), sha256=file_hash(path), image_ids=sorted(found)))
    # Research-side adjudication/case records sometimes precede raw manifests.
    records = WORKTREE / 'research/investigations/qwen3-vl-dense-enumeration'
    for path in sorted(records.rglob('*.json')):
        if '2026-09-12-parallel-owner-research' in str(path):
            continue
        found = image_ids(read(path))
        excluded.update(found)
        sources.append(dict(path=str(path), sha256=file_hash(path), image_ids=sorted(found)))
    rows = [json.loads(line) for line in SOURCE.open()]
    universe = [r['image_id'] for r in rows]
    require(len(universe) == len(set(universe)), 'duplicate source image identity')
    selected = select_ids(universe, excluded)
    result = dict(schema='parallel_transfer.selection.v1', status='frozen_before_new_outputs',
        source=dict(path=str(SOURCE), sha256=file_hash(SOURCE)), salt=SALT,
        source_images=len(universe), excluded_image_ids=sorted(excluded),
        excluded_source_images=len(set(universe) & excluded),
        eligible_images=len(set(universe) - excluded), image_ids=selected,
        image_ids_sha256=digest(selected), exposure_sources=sources,
        provenance_boundary='Disjoint from identity-bearing investigation input, selection, run and visual manifests plus research JSON records at freeze; no exhaustive raw-log or outside-investigation audit, no pretraining/SFT-disjointness claim.',
        selection_rule='SHA256(salt + decimal image ID), first256 among source minus exclusions; no outputs, object counts or backfill')
    publish(ROOT / 'selection.json', result)
    print(json.dumps({k:result[k] for k in ('status','source_images','excluded_source_images','eligible_images','image_ids_sha256')}))


def binding(path):
    return dict(path=str(path), sha256=file_hash(path))


def rebase_case(record):
    record['case']['input_record']['images']=[os.path.relpath(record['case']['image_path'],NATIVE_SOURCE.parent)]
    return record


def first_row_prefix(ids, divergence, box_end, eos=151645):
    """Close the donor row containing first divergence; EOS is terminal, never invented."""
    stop=next((i+1 for i in range(divergence,len(ids)) if ids[i] in (box_end,eos)),None)
    require(stop is not None,'divergent history has neither complete row nor EOS; secondary infeasible')
    return ids[:stop],ids[stop:]


def cross_jobs(d,tokenizer):
    consumers={label:{r['image_id']:r for r in read(path)} for label,path in [
        ('C32',Path(d['sources']['C_consumer']['path'])),('D17',PRIOR/'endpoint-D/consumer.json')]}
    close=tokenizer.convert_tokens_to_ids('<|box_end|>')
    jobs=[]
    for iid in (25274,511251,417044):
        c,drow=consumers['C32'][iid],consumers['D17'][iid]
        a,b=c['action_ids'],drow['action_ids']
        divergence=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),min(len(a),len(b)))
        require(divergence < max(len(a),len(b)),'identical histories; no cross boundary')
        for donor in ('C32','D17'):
            row=consumers[donor][iid]
            prefix,suffix=first_row_prefix(row['action_ids'],divergence,close)
            jobs.append(dict(image_id=iid,example_id=row['example_id'],donor=donor,
                divergence_token_index=divergence,prefix_ids=prefix,expected_self_suffix=suffix,
                remaining_budget=CAP-len(prefix),terminal=prefix[-1]==151645,
                history_source=binding(Path(d['sources']['C_consumer']['path']) if donor=='C32' else PRIOR/'endpoint-D/consumer.json')))
    return jobs


def prepare():
    from src.config.inference import InferConfig
    from src.data.examples import raw_example_from_jsonl_row
    from probes.dora_owner_learning.runtime import build_request
    from src.qwen.native import prepare_native_inputs
    from src.qwen.runtime_loading import QwenLoadOptions, load_qwen_components_from_options
    from src.adapters.dora import inspect_dora_adapter_payload
    selection = read(ROOT / 'selection.json')
    require(file_hash(SOURCE) == selection['source']['sha256'], 'source bytes changed')
    d = read(PRIOR / 'endpoint-preparation/packet.json')
    cpath = Path(d['sources']['C_packet']['path'])
    require(file_hash(cpath) == d['sources']['C_packet']['sha256'], 'accepted C packet changed')
    c = read(cpath)
    anchors = dict(Stable50=d['model']['current_adapter'], C32=c['candidate']['saved_adapter'],
                   D17=d['candidate']['saved_adapter'])
    retained_anchor_identities=copy.deepcopy(anchors)
    for label, adapter in list(anchors.items()):
        actual=inspect_dora_adapter_payload(adapter['root'], d['model']['base_model_path'])
        # The accepted Stable50 receipt predates coordexp-swift -> coordexp-infras
        # metadata naming. Require every payload/semantic field exactly, not hash aliasing.
        require({k:v for k,v in actual.items() if k not in ('version','fingerprint')} ==
                {k:v for k,v in adapter.items() if k not in ('version','fingerprint')},f'{label} payload identity changed')
        require(adapter['version'] in ('coordexp-swift-dora-adapter-v1','coordexp-infras-dora-adapter-v1')
                and actual['version']=='coordexp-infras-dora-adapter-v1','unsupported identity version')
        anchors[label]=actual
    require(len({a['fingerprint'] for a in anchors.values()}) == 3, 'checkpoint labels alias')
    config = copy.deepcopy(d['config'])
    config['data']['input_jsonl'] = str(NATIVE_SOURCE)
    infer_config=InferConfig.model_validate(config)
    qwen = load_qwen_components_from_options(QwenLoadOptions(
        base_model=d['model']['base_model_path'], dtype='fp32', attn_implementation='sdpa',
        patch_embed_linearization='enabled', load_model=False))
    all_rows = {r['image_id']:(n,r) for n,line in enumerate(NATIVE_SOURCE.open()) for r in [json.loads(line)]}
    source_rows={r['image_id']:r for line in SOURCE.open() for r in [json.loads(line)]}
    records = []
    for iid in selection['image_ids']:
        index, row = all_rows[iid]
        original=source_rows[iid]
        require(sorted(map(digest,row['objects']))==sorted(map(digest,original['objects'])) and
                (NATIVE_SOURCE.parent/row['images'][0]).resolve()==(SOURCE.parent/original['images'][0]).resolve() and
                (row['width'],row['height'])==(original['width'],original['height']),'native serialization changed source image/GT')
        raw = raw_example_from_jsonl_row(row, jsonl_path=NATIVE_SOURCE, row_number=index+1, raw_line=json.dumps(row))
        request,image_plan,_=build_request(raw,config=infer_config,qwen=qwen,row_index=index)
        plan=image_plan.to_artifact_dict()
        batch = prepare_native_inputs(qwen.processor,[request],device='cpu',record_media_identity=True)
        plan.update(observed_image_grid_thw=list(batch.image_grids[0]), executed_media_sha256=batch.media_sha256[0],
                    backend_prompt_token_count=len(batch.prompt_token_ids[0]), backend_projection_evidence_kind='hf_executed_tensors')
        case=dict(row_id=str(raw.example_id),row_index=index,image_path=str(raw.image.path),
            image_width=raw.image.width,image_height=raw.image.height,input_record=row,image_plan=plan)
        golden={k:case[k] for k in ('row_id','row_index','image_path','image_width','image_height')}
        golden.update(example_id=str(raw.example_id), gt=[obj.to_artifact_dict() for obj in raw.objects])
        records.append(dict(example_id=str(raw.example_id),image_id=iid,split='fresh256',case=case,
                            golden=golden,prompt_token_ids=list(batch.prompt_token_ids[0])))
    # Old exposed cases qualify checkpoint/native consumer without rereading fresh images.
    qualification = copy.deepcopy(d['eval_records'][:2])
    for record in qualification:
        rebase_case(record)
    expected = {'Stable50':{r['example_id']:r['stable_ids'] for r in qualification}}
    for label,path in [('C32',Path(d['sources']['C_consumer']['path'])),('D17',PRIOR/'endpoint-D/consumer.json')]:
        expected[label] = {r['example_id']:r['action_ids'] for r in read(path) if r['example_id'] in expected['Stable50']}
    secondary_records=copy.deepcopy([r for r in d['eval_records'] if r['image_id'] in (25274,511251,417044)])
    for record in secondary_records:
        rebase_case(record)
    packet=dict(schema='parallel_transfer.packet.v1', labels=list(LABELS), anchors=anchors,retained_anchor_identities=retained_anchor_identities, config=config,
        model=d['model'], records=records, qualification_records=qualification, qualification_expected=expected,
        cross_records=secondary_records,cross_jobs=cross_jobs(d,qwen.tokenizer),
        native_source=binding(NATIVE_SOURCE),native_source_parity='Exact selected IDs, image paths/dimensions and whole owner-object multisets; only accepted geo_sorted_xy serialization differs from selection source',
        selection=binding(ROOT/'selection.json'), sources=[binding(PRIOR/'root-acceptance.json'),binding(PRIOR/'endpoint-preparation/packet.json'),binding(cpath)],
        producer=binding(Path(__file__).resolve()), cap=CAP, physical_gpus=[6,7],
        resources=dict(natural_continuations=768, qualification_continuations=6, secondary_cells=12,total_action_token_upper_bound=768*CAP,
                       model_loads_full=6, model_loads_qualification=3, peak_cuda_estimate_gib=16,
                       peak_rss_estimate_gib=18, estimated_wall_minutes=[35,90]))
    publish(ROOT/'packet.json',packet)
    print(json.dumps(dict(status='prepared',records=len(records),packet_sha256=file_hash(ROOT/'packet.json'))))


def validate_population(rows, records, label):
    require(label in LABELS, 'unknown checkpoint label')
    expected=[r['example_id'] for r in records]
    require([r['example_id'] for r in rows] == expected and len(set(expected)) == len(rows), 'ordered population mismatch')
    require(all(r['arm']==label and r['prefix_ids']==r['forced_ids']==[] and r['remaining_budget']==CAP for r in rows),
            'label or natural intervention mismatch')


def execute(label, shard, phase):
    import torch
    from probes.dora_owner_learning.runtime import load_policy
    from probes.dora_owner_learning.route_access import checkpoint_config
    from probes.source_rweak_row_cross.run import build_requests, native_record
    from src.adapters.dora import inspect_dora_adapter_payload
    from src.config.inference import InferConfig
    from src.qwen.native import prepare_native_inputs
    from src.qwen.generation import NativeGenerationPolicy,generate_continuations
    packet=read(ROOT/'packet.json')
    require(file_hash(Path(__file__).resolve())==packet['producer']['sha256'],'producer changed after packet')
    require(label in LABELS and shard in (0,1), 'unknown label/shard')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')==str(6+shard) and torch.cuda.device_count()==1,'reserved single GPU only')
    records=packet['qualification_records'] if phase=='qualify' else packet['records'][shard::2]
    jobs=[None]*len(records)
    if phase=='cross':
        require(label in ('C32','D17'),'secondary has no Stable50 arm')
        jobs=packet['cross_jobs'][shard::2]
        lookup={r['example_id']:r for r in packet['cross_records']}
        records=[lookup[j['example_id']] for j in jobs]
    adapter=packet['anchors'][label]
    require(inspect_dora_adapter_payload(adapter['root'],packet['model']['base_model_path'])==adapter,'actual anchor changed')
    run=ROOT/phase/f'{label}-shard-{shard}'
    run.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    terminal=dict(status='running',arm=label,shard=shard,phase=phase,pid=os.getpid(),
                  packet_sha256=file_hash(ROOT/'packet.json'),adapter_fingerprint=adapter['fingerprint'],
                  model_loads=0,continuations=0,new_tokens=0)
    publish(run/'launch.json',dict(terminal))
    counters={'model':0,'image':0}
    handles=[]
    try:
        config=checkpoint_config(InferConfig.model_validate(packet['config']),adapter['root'])
        torch.cuda.reset_peak_memory_stats()
        qwen,identity=load_policy(config,device=torch.device('cuda:0'))
        terminal['model_loads']=1
        live=identity['model_identity']['adapter']
        require(live['adapter_path']==adapter['root'] and live['merged_adapters']==[],'wrong/merged anchor')
        require(identity['effective_settings']['observed_model_dtype']['parameter_dtype_names']==['torch.float32'] and
                identity['effective_settings']['observed_attn_implementation']=='sdpa','native FP32 SDPA changed')
        publish(run/'model.json',dict(identity=identity,arm=label,adapter=adapter))
        qwen.model.eval()
        for p in qwen.model.parameters(): p.requires_grad_(False)
        def count_model(*_): counters['model']+=1
        def count_image(*_): counters['image']+=1
        handles.append(qwen.model.register_forward_pre_hook(count_model))
        visual=[m for n,m in qwen.model.named_modules() if n.endswith('visual')]
        require(len(visual)==1,'visual hook ambiguity')
        handles.append(visual[0].register_forward_pre_hook(count_image))
        policy=NativeGenerationPolicy(temperature=0.0,top_p=1.0,top_k=0,repetition_penalty=1.0,use_model_defaults=False)
        with (run/'rows.jsonl').open('x') as stream:
            for frozen,job in zip(records,jobs,strict=True):
                requests,_=build_requests(qwen,packet['config'],[frozen['case']])
                batch=prepare_native_inputs(qwen.processor,requests,device='cuda:0',record_media_identity=True)
                plan=frozen['case']['image_plan']
                require(list(batch.prompt_token_ids[0])==frozen['prompt_token_ids'] and batch.media_sha256[0]==plan['executed_media_sha256']
                        and list(batch.image_grids[0])==plan['observed_image_grid_thw'],'frozen input identity changed')
                prefix=[] if job is None else job['prefix_ids']
                budget=CAP-len(prefix)
                if job is not None and job['terminal']:
                    free=[];stop='im_end';request_id=frozen['example_id']
                else:
                    with torch.inference_mode():
                        generated=generate_continuations(qwen.model,batch,extensions=[prefix],budgets=[budget],eos_token_id=151645,
                            pad_token_id=qwen.tokenizer.pad_token_id,policy=policy,trace='none')[0]
                    free=list(generated.token_ids);stop=generated.stop_reason;request_id=generated.request_id
                    accepted._checked_action(free,stop,budget)
                ids=prefix+free
                text=qwen.tokenizer.decode(ids,skip_special_tokens=False)
                parsed=native_record(text,frozen['case'],frozen['golden'],stop)
                observed=dict(prompt_token_ids_sha256=digest(frozen['prompt_token_ids']),executed_media_sha256=batch.media_sha256[0],
                              observed_image_grid_thw=list(batch.image_grids[0]))
                row=dict(schema='parallel_transfer.natural.v1',arm=label,shard=shard,example_id=frozen['example_id'],
                    image_id=frozen['image_id'],split=frozen['split'],request_id=request_id,action_ids=ids,prefix_ids=prefix,forced_ids=prefix,
                    remaining_budget=budget,text=text,stop_reason=stop,parsed=parsed,
                    score=score(parsed,seed=None,length=len(ids),stop=stop),overlap_counts=overlap_counts(parsed),
                    batch_identity_sha256=digest(observed),packet_sha256=terminal['packet_sha256'],adapter_fingerprint=adapter['fingerprint'],**observed)
                if job is not None:
                    row.update(schema='parallel_transfer.cross.v1',donor=job['donor'],free_ids=free,terminal_prefix=job['terminal'])
                    row.update(cross_credit=cross_credit(row,frozen,qwen.tokenizer))
                    if label==job['donor']:require(free==job['expected_self_suffix'],'self-history exact remaining-token parity')
                stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush();os.fsync(stream.fileno())
                terminal['continuations']+=1;terminal['new_tokens']+=len(free)
                if phase=='qualify': require(ids==packet['qualification_expected'][label][frozen['example_id']], 'qualification archived natural token parity')
        terminal.update(status='completed',exit_code=0)
    except BaseException as exc:
        terminal.update(status='failed',exit_code=1,error=repr(exc),traceback=traceback.format_exc())
        raise
    finally:
        for handle in handles: handle.remove()
        terminal.update(elapsed_seconds=time.monotonic()-started,model_forwards=counters['model'],image_forwards=counters['image'],
                        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0,
                        peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0,
                        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        publish(run/'terminal.json',terminal)


def launch(phase):
    out=ROOT/phase
    require(not out.exists(),'occupied phase; no automatic relaunch')
    out.mkdir(parents=True)
    # Two long independent lanes, no NCCL/process group and no eight-rank assumption.
    processes=[]
    for shard in ([0] if phase=='qualify' else [0,1]):
        command=[sys.executable,'-m','probes.parallel_owner_research.transfer','worker','--phase',phase,'--shard',str(shard)]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(6+shard),OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
        log=(out/f'shard-{shard}.log').open('x')
        processes.append((shard,subprocess.Popen(command,env=env,cwd=WORKTREE,stdout=log,stderr=subprocess.STDOUT),log))
    exits=[]
    for shard,process,log in processes:
        code=process.wait();log.close();exits.append(dict(shard=shard,exit_code=code))
    publish(out/'outer-exits.json',exits)
    require(all(r['exit_code']==0 for r in exits),'producer failure; preserve partial outputs')


def paired_summary(before,after):
    import numpy as np
    require([r['example_id'] for r in before]==[r['example_id'] for r in after],'unpaired image order')
    pairs=[]
    for a,b in zip(before,after,strict=True):
        pairs.append(dict(image_id=a['image_id'],example_id=a['example_id'],owner_changes=accepted.owner_changes(a['score'],b['score']),
            delta={t:{k:b['score'][t][k]-a['score'][t][k] for k in ('tp','fp','fn','f1')} for t in ('50','60','80')}))
    rng=np.random.default_rng(20260912)
    index=rng.integers(0,len(pairs),size=(10000,len(pairs)))
    uncertainty={}
    for threshold in ('50','60','80'):
        left=np.array([[r['score'][threshold][k] for k in ('tp','fp','fn')] for r in before],dtype=float)
        right=np.array([[r['score'][threshold][k] for k in ('tp','fp','fn')] for r in after],dtype=float)
        aa,bb=left[index].sum(1),right[index].sum(1)
        def f1(x): return np.divide(2*x[:,0],2*x[:,0]+x[:,1]+x[:,2],out=np.zeros(len(x)),where=(2*x[:,0]+x[:,1]+x[:,2])>0)
        delta=right[:,0]-left[:,0]
        uncertainty[threshold]=dict(delta_tp95=np.quantile(bb[:,0]-aa[:,0],[.025,.975]).tolist(),
            delta_micro_f1_95=np.quantile(f1(bb)-f1(aa),[.025,.975]).tolist(),
            positive_images=int((delta>0).sum()),negative_images=int((delta<0).sum()),tied_images=int((delta==0).sum()),
            largest_absolute_tp_changes=sorted([dict(image_id=p['image_id'],delta_tp=int(v)) for p,v in zip(pairs,delta,strict=True)],
                                              key=lambda r:(-abs(r['delta_tp']),r['image_id']))[:20])
    return dict(owner_counts=accepted.owner_change_counts([r['score'] for r in before],[r['score'] for r in after]),
                paired_image_bootstrap=dict(replicates=10000,seed=20260912,intervals=uncertainty),images=pairs)


def burden(rows):
    value=accepted.burden(rows)
    value['parsed_plus_dropped_rows']=value['row_starts']
    value['row_starts']=sum(row['text'].count('<|object_ref_start|>') for row in rows)
    value['row_start_accounting_difference']=value['row_starts']-value['parsed_plus_dropped_rows']
    return value


def cross_credit(row,frozen,tokenizer):
    from probes.source_rweak_row_cross.run import native_record
    prefix=native_record(tokenizer.decode(row['prefix_ids'],skip_special_tokens=False),frozen['case'],frozen['golden'],'conditional')
    suffix=native_record(tokenizer.decode(row['free_ids'],skip_special_tokens=False),frozen['case'],frozen['golden'],row['stop_reason'])
    p=score(prefix,seed=None,length=len(row['prefix_ids']),stop='conditional')
    s=score(suffix,seed=None,length=len(row['free_ids']),stop=row['stop_reason'])
    return dict(forced_prefix_score=p,free_suffix_score=s,
                new_free_owners={t:sorted(set(s[t]['owners'])-set(p[t]['owners'])) for t in ('50','60','80')},
                credit_boundary='Only free suffix predictions enter free global matching; forced rows never get free credit. Full-output score remains intervention outcome.')


def merge(phase):
    from transformers import AutoTokenizer
    packet=read(ROOT/'packet.json')
    tokenizer=AutoTokenizer.from_pretrained(packet['model']['base_model_path'],local_files_only=True)
    if phase=='cross':
        from probes.source_rweak_row_cross.run import native_record
        rows=[];terminals=[]
        require(all(r['exit_code']==0 for r in read(ROOT/phase/'outer-exits.json')),'cross outer failure')
        frozen={r['example_id']:r for r in packet['cross_records']}
        for label in ('C32','D17'):
            for shard in (0,1):
                path=ROOT/phase/f'{label}-shard-{shard}'
                terminal=read(path/'terminal.json');terminals.append(terminal)
                require(terminal['status']=='completed' and terminal['continuations']==3,'cross incomplete worker')
                for row,job in zip(accepted.load_jsonl(path/'rows.jsonl'),packet['cross_jobs'][shard::2],strict=True):
                    require(row['arm']==label and row['donor']==job['donor'] and row['example_id']==job['example_id'],'cross job association')
                    require(row['prefix_ids']==row['forced_ids']==job['prefix_ids'] and row['action_ids']==row['prefix_ids']+row['free_ids'],'cross token boundary')
                    require(row['remaining_budget']==job['remaining_budget'] and len(row['action_ids'])<=CAP,'cross cap')
                    require(row['adapter_fingerprint']==packet['anchors'][label]['fingerprint'] and row['packet_sha256']==file_hash(ROOT/'packet.json'),'cross model identity')
                    require(tokenizer.decode(row['action_ids'],skip_special_tokens=False)==row['text'],'cross token/text')
                    parsed=native_record(row['text'],frozen[row['example_id']]['case'],frozen[row['example_id']]['golden'],row['stop_reason'])
                    require(parsed==row['parsed'] and cross_credit(row,frozen[row['example_id']],tokenizer)==row['cross_credit'],'cross cold parser/credit')
                    if label==job['donor']:require(row['free_ids']==job['expected_self_suffix'],'self remaining token parity')
                    rows.append(row)
        require(len(rows)==12 and len({(r['arm'],r['donor'],r['image_id']) for r in rows})==12,'exact12 cross')
        publish(ROOT/phase/'consumer.json',rows)
        publish(ROOT/phase/'result.json',dict(schema='parallel_transfer.cross_result.v1',status='cold_verified',cells=12,
            diagonal_exact_suffix_parity=True,rows=[{k:r[k] for k in ('arm','donor','image_id','score','cross_credit','terminal_prefix')} for r in rows],
            resources=dict(terminals=terminals,allocated_gpu_seconds=sum(r['elapsed_seconds'] for r in terminals)),
            boundary='Secondary3 exposed images; no fresh denominator, no forced-row autonomous credit. Terminal donor prefix is absorbing EOS and produces no free tokens.'))
        print(json.dumps(dict(status='cold_verified',phase=phase,cells=12)))
        return
    rows_by_label={};terminals=[]
    records=packet['qualification_records'] if phase=='qualify' else packet['records']
    require(all(r['exit_code']==0 for r in read(ROOT/phase/'outer-exits.json')),'outer process failure')
    for label in LABELS:
        rows=[]
        for shard in ([0] if phase=='qualify' else [0,1]):
            path=ROOT/phase/f'{label}-shard-{shard}'
            terminal=read(path/'terminal.json');terminals.append(terminal)
            require(terminal['status']=='completed' and terminal['exit_code']==0,'incomplete actual worker')
            rows.extend(accepted.load_jsonl(path/'rows.jsonl'))
        lookup={r['example_id']:r for r in rows}
        require(len(lookup)==len(rows)==len(records),'duplicate/missing consumer records')
        rows=[lookup[r['example_id']] for r in records]
        validate_population(rows,records,label)
        for row,frozen in zip(rows,records,strict=True):
            require(row['adapter_fingerprint']==packet['anchors'][label]['fingerprint'] and row['packet_sha256']==file_hash(ROOT/'packet.json'),'row checkpoint/packet mismatch')
            accepted.old_endpoint()._cold_natural(row,frozen,tokenizer)
            if phase=='qualify':require(row['action_ids']==packet['qualification_expected'][label][row['example_id']],'qualification parity')
        rows_by_label[label]=rows
    result=dict(schema='parallel_transfer.result.v1',phase=phase,status='cold_verified',images=len(records),
        packet=binding(ROOT/'packet.json'),selection=binding(ROOT/'selection.json'),
        summaries={label:dict(score=aggregate_scores([r['score'] for r in rows]),burden=burden(rows)) for label,rows in rows_by_label.items()},
        comparisons={f'{after}_vs_{before}':paired_summary(rows_by_label[before],rows_by_label[after])
                     for before,after in [('Stable50','C32'),('Stable50','D17'),('D17','C32')]},
        resources=dict(terminals=terminals,allocated_gpu_seconds=sum(r['elapsed_seconds'] for r in terminals),
                       model_forwards=sum(r['model_forwards'] for r in terminals),image_forwards=sum(r['image_forwards'] for r in terminals)),
        uncertainty_boundary='Image-paired percentile bootstrap over frozen eligible-source sample; no full-corpus, complete-scene or pretraining-disjointness claim')
    publish(ROOT/phase/'consumer.json',rows_by_label)
    publish(ROOT/phase/'result.json',result)
    print(json.dumps(dict(status=result['status'],phase=phase,images=len(records),result_sha256=file_hash(ROOT/phase/'result.json'))))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['select','prepare','launch','worker','merge'])
    parser.add_argument('--phase',choices=['qualify','full','cross'],default='qualify')
    parser.add_argument('--shard',type=int,default=0)
    args=parser.parse_args()
    if args.command=='select':freeze_selection()
    elif args.command=='prepare':prepare()
    elif args.command=='launch':launch(args.phase)
    elif args.command=='merge':merge(args.phase)
    else:
        # A new process per checkpoint makes model-load and allocator accounting real.
        for label in (('C32','D17') if args.phase=='cross' else LABELS):
            command=[sys.executable,'-c',f'from probes.parallel_owner_research.transfer import execute; execute({label!r},{args.shard},{args.phase!r})']
            subprocess.run(command,check=True,cwd=WORKTREE)


if __name__=='__main__':main()
