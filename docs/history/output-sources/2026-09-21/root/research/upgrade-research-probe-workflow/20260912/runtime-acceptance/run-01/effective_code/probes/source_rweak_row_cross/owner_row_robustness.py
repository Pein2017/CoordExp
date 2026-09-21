"""Frozen owner-row eligibility and native conditional continuation probe."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import resource
import signal
import time
import traceback

from probes.dora_owner_learning.candidate_opportunity import (
    digest, file_hash, indexed, require, rows, score,
)
from probes.dora_owner_learning.round1_realization import SOURCE_ROOT, CANDIDATE_ROOT, UPDATE_ROOT
from probes.dora_owner_learning.reward_rows import _gt_objects, _pred_objects
from src.data.geometry import iou_xyxy
from .run import native_record, load_components, build_requests

OUTPUT = SOURCE_ROOT.parents[2] / '2026-09-10-owner-row-continuation-robustness'
CAP = 3084


def publish(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


def native_rows(ids, tokenizer):
    """Reject malformed or non-native row layouts, never rebuild row IDs."""
    start = tokenizer.token_to_id('<|object_ref_start|>')
    end = tokenizer.token_to_id('<|object_ref_end|>')
    box = tokenizer.token_to_id('<|box_start|>')
    close = tokenizer.token_to_id('<|box_end|>')
    coords = {tokenizer.token_to_id(f'<|coord_{i}|>'): i for i in range(1000)}
    result, pos = [], 0
    while pos < len(ids) and ids[pos] != 151645:
        require(ids[pos] == start, 'non-native row start')
        stop = ids.index(close, pos) + 1
        chunk = ids[pos:stop]
        boundary = chunk.index(end)
        require(boundary > 1 and chunk[boundary+1] == box and len(chunk) == boundary+7,
                'non-native row layout')
        positions = list(range(boundary+2, boundary+6))
        require(all(chunk[i] in coords for i in positions), 'invalid coordinate IDs')
        xyxy = [coords[chunk[i]] for i in positions]
        require(xyxy[0] < xyxy[2] and xyxy[1] < xyxy[3], 'invalid box')
        result.append(dict(start=pos, stop=stop, ids=chunk, description=chunk[1:boundary],
                           coords=xyxy, positions=positions))
        pos = stop
    require(ids[pos:] in ([], [151645]), 'tokens after EOS')
    return result


def variant(row, tokenizer):
    xyxy = row['coords'][:]
    xyxy[2] += 1 if xyxy[2] < 999 else -1
    require(xyxy[2] > xyxy[0], 'no legal one-bin x2 variant')
    ids = row['ids'][:]
    ids[row['positions'][2]] = tokenizer.token_to_id(f'<|coord_{xyxy[2]}|>')
    return dict(row, ids=ids, coords=xyxy)


def branch(case, arm):
    prefix = case['sample_prefix_ids'] if arm == 'sample_A' else case['prefix_ids']
    forced = case[{'B':'B', 'partial_A':'A', 'A':'A', 'Aprime':'Aprime', 'sample_A':'A'}[arm]]['ids']
    if arm == 'partial_A':
        first = next(i for i, (a,b) in enumerate(zip(forced, case['B']['ids'])) if a != b)
        require(first in case['A']['positions'], 'first difference is not a coordinate')
        forced = forced[:first+1]
    require(0 < CAP-len(prefix)-len(forced), 'nonpositive remaining budget')
    return dict(prefix=prefix, forced=forced, remaining=CAP-len(prefix)-len(forced))


def validate_admission(manifest, admission):
    require(admission['manifest_sha256'] == digest(manifest), 'admission manifest mismatch')
    ids = admission['admitted_case_ids']
    require(len(ids)==len(set(ids)), 'duplicate admitted IDs')
    cases = indexed(manifest['selected'], 'case_id')
    require(set(ids) <= cases.keys(), 'missing admitted ID')
    require(len(ids) <= 12, 'panel cap')
    return [cases[i] for i in ids]


def census(output):
    from tokenizers import Tokenizer
    output.mkdir(parents=True, exist_ok=False)
    plan_path = UPDATE_ROOT/'round-1/plan.json'
    plan = json.loads(plan_path.read_text())
    tok = Tokenizer.from_file(plan['model']['base_model_path']+'/tokenizer.json')
    candidates = json.loads((CANDIDATE_ROOT/'cases.json').read_text())
    raw = indexed(rows(SOURCE_ROOT/'gt_vs_pred.jsonl'), 'row_id')
    groups = indexed(plan['population']['groups'], 'example_id')
    traces = defaultdict(list)
    for t in rows(SOURCE_ROOT/'pred_token_trace.jsonl'):
        if t['trace_type']=='generated_token' and not t['is_pad']:
            traces[t['row_id']].append(t)
    census_rows, eligible = [], []
    for candidate in candidates:
        strong = [s for s in candidate['samples'] if s['comparison']['strong_joint_witness']]
        owners = sorted({o for s in strong for o in s['comparison']['gained']})
        if not owners: continue
        eid = candidate['example_id']; golden = raw[eid]
        trace = sorted(traces[eid], key=lambda t:t['generated_step_index'])
        source_ids = [t['token_id'] for t in trace]
        require(tok.decode(source_ids, skip_special_tokens=False)==golden['raw_decode_text'], 'Source token identity')
        try:
            source_rows = native_rows(source_ids, tok)
            require(len(source_rows)==len(golden['pred']), 'Source row/parser association')
            require(candidate['greedy']['invalid_predictions']==0,'Source projection/row association')
        except ValueError as exc:
            census_rows.extend(dict(example_id=eid, owner=o, failures={str(exc):1}) for o in owners)
            continue
        gt = _gt_objects(golden, row_id=eid); preds,_ = _pred_objects(golden)
        source_matches = {m['pred_index']:m for m in candidate['greedy']['50']['matches']}
        for owner in owners:
            failures = Counter(); found = []
            gi = next(i for i,g in enumerate(golden['gt']) if str(g['object_id'])==owner)
            far = max((iou_xyxy(gt[gi][1], p[1]) for p in preds), default=0)<.1
            for sample in sorted(strong, key=lambda s:s['seed']):
                if owner not in sample['comparison']['gained']: continue
                action = next(a for a in groups[eid]['actions'] if a['seed']==sample['seed'])
                try:
                    sample_rows = native_rows(action['action_token_ids'], tok)
                    match = next(m for m in sample['50']['matches'] if m['owner']==owner)
                    A = sample_rows[match['pred_index']]; Ap = variant(A,tok)
                    parsed_ap = native_record(tok.decode(Ap['ids'],skip_special_tokens=False), {'row_id':eid},golden,'conditional')
                    ap_preds, bad = _pred_objects(parsed_ap)
                    require(not bad and len(ap_preds)==1 and ap_preds[0][0]==gt[gi][0]
                            and iou_xyxy(ap_preds[0][1],gt[gi][1])>=.5, 'variant owner IoU50')
                except (ValueError, IndexError) as exc:
                    failures[str(exc)] += 1; continue
                for bi,B in enumerate(source_rows):
                    reason = None
                    if B['description'] != A['description']: reason='description mismatch'
                    elif B['positions'] != A['positions'] or len(B['ids'])!=len(A['ids']): reason='row layout mismatch'
                    elif bi not in source_matches or source_matches[bi]['owner']==owner: reason='B lacks distinct owner'
                    elif any(tuple(source_rows[j]['coords'][:2])>=tuple(source_rows[j+1]['coords'][:2]) for j in range(bi)): reason='early Source order anomaly'
                    elif not ((bi==0 or tuple(source_rows[bi-1]['coords'][:2])<tuple(A['coords'][:2])) and tuple(A['coords'][:2])<tuple(B['coords'][:2])): reason='A outside strict insertion interval'
                    elif any(preds[j][0]==gt[gi][0] and iou_xyxy(preds[j][1],gt[gi][1])>=.5 for j in range(bi)): reason='prior A direct binding'
                    if reason: failures[reason]+=1; continue
                    item=dict(case_id=f'{eid}:{owner}', example_id=eid,image_id=candidate['image_id'],owner=owner,
                        gt_index=gi, seed=sample['seed'], far_any_iou_lt_point1=far, boundary=bi,
                        B_owner=source_matches[bi]['owner'], A=A, Aprime=Ap, B=B,
                        prefix_ids=source_ids[:B['start']], sample_prefix_ids=action['action_token_ids'][:A['start']],
                        source_ids=source_ids, golden=golden, group=groups[eid])
                    for arm in ('B','partial_A','A','Aprime','sample_A'): branch(item,arm)
                    found.append(item)
            found.sort(key=lambda c:(c['seed'],c['boundary']))
            census_rows.append(dict(example_id=eid,owner=owner, far_any_iou_lt_point1=far,
                                    eligible_tuples=len(found),failures=dict(failures)))
            if found: eligible.append(found[0])
    require(len(census_rows)==65, 'strong distinct owner denominator differs')
    selected=[]; used=set(); counts=Counter()
    for c in eligible:
        c['rank_key']=[int(c['image_id']),int(c['owner']),c['seed'],c['boundary']]
    for c in sorted(eligible,key=lambda c:c['rank_key']):
        stratum=c['far_any_iou_lt_point1']
        if c['image_id'] not in used and counts[stratum]<6:
            selected.append(c);used.add(c['image_id']);counts[stratum]+=1
    manifest=dict(schema='owner_row_robustness.v1', selected=selected, census=census_rows,
        denominator=65, eligible_owner_pairs=len(eligible), selected_count=len(selected),
        ranking='numeric COCO image ID, numeric GT annotation owner ID, seed, boundary',
        eligible_rank_keys=[c['rank_key'] for c in sorted(eligible,key=lambda c:c['rank_key'])],
        source_files={str(p):file_hash(p) for p in (plan_path,CANDIDATE_ROOT/'cases.json',SOURCE_ROOT/'gt_vs_pred.jsonl',SOURCE_ROOT/'pred_token_trace.jsonl')})
    publish(output/'manifest.json',manifest)
    review_cards(output,selected)
    publish(output/'census.json',{k:v for k,v in manifest.items() if k not in ('selected','source_files')})
    print(json.dumps(dict(denominator=65,eligible=len(eligible),selected=len(selected),strata=dict(counts),manifest_sha256=digest(manifest))))


def review_cards(output, cases):
    from PIL import Image, ImageDraw
    for c in cases:
        raw=Image.open(c['golden']['image_path']).convert('RGB')
        annotated=raw.copy(); draw=ImageDraw.Draw(annotated)
        w,h=raw.size
        def box(coords,color,label):
            xy=[coords[0]*w/999,coords[1]*h/999,coords[2]*w/999,coords[3]*h/999]
            draw.rectangle(xy,outline=color,width=2);draw.text((xy[0],xy[1]),label,fill=color)
            return xy
        for j,p in enumerate(c['golden']['pred'][:c['boundary']]):
            draw.rectangle(p['bbox'],outline='cyan',width=1)
            draw.text(tuple(p['bbox'][:2]),f'P{j}',fill='cyan')
        box(c['golden']['gt'][c['gt_index']]['bbox'],'yellow','GT A')
        a=box(c['A']['coords'],'lime','A');box(c['Aprime']['coords'],'magenta',"A'")
        b=box(c['B']['coords'],'red','B')
        region=(max(0,int(min(a[0],b[0])-40)),max(0,int(min(a[1],b[1])-40)),
                min(w,int(max(a[2],b[2])+40)),min(h,int(max(a[3],b[3])+40)))
        panels=[raw,annotated,raw.crop(region),annotated.crop(region)]
        canvas=Image.new('RGB',(max(p.width for p in panels)*2,max(p.height for p in panels)*2+35),'white')
        cw,ch=canvas.width//2,(canvas.height-35)//2
        for i,p in enumerate(panels): canvas.paste(p,((i%2)*cw,35+(i//2)*ch))
        ImageDraw.Draw(canvas).text((4,5),f"{c['case_id']} seed {c['seed']} boundary {c['boundary']} green A magenta A' red B cyan prior",fill='black')
        canvas.save(output/(c['case_id'].replace(':','_')+'.png'))


def incidence(parsed, start=0, threshold=.5, category_consistent=True):
    gt=_gt_objects(parsed,row_id=parsed['row_id'])
    pred=[(j,projected[0]) for j,obj in enumerate(parsed['pred'])
          if (projected:=_pred_objects(dict(parsed,pred=[obj]))[0])]
    return {str(parsed['gt'][i]['object_id']): [j for j,p in pred if j>=start and
            (not category_consistent or p[0]==g[0]) and iou_xyxy(p[1],g[1])>=threshold] for i,g in enumerate(gt)}


def consume(path, manifest, tokenizer, expected):
    """Disk-native parser/owner consumer, including partial and malformed outputs."""
    records=rows(path); seen=set(); cases=indexed(manifest['selected'],'case_id')
    for record in records:
        key=(record['case_id'],record['arm']);require(key not in seen,'duplicate output IDs');seen.add(key)
        require(record['manifest_sha256']==digest(manifest),'output manifest identity')
        case=cases[record['case_id']];job=branch(case,record['arm'])
        ids=job['prefix']+job['forced']+record['suffix_ids']
        require(ids==record['action_ids'] and len(ids)<=CAP and len(record['suffix_ids'])<=job['remaining'], 'output budget/identity')
        text=tokenizer.decode(ids,skip_special_tokens=False)
        require(text==record['text'],'output text identity')
        parsed=native_record(text,{'row_id':case['example_id']},case['golden'],record['stop_reason'])
        require(parsed==record['parsed'],'disk native parser differs')
        record['score']=score(parsed,seed=-1,length=len(ids),stop=record['stop_reason'])
        projection=[j for j,obj in enumerate(parsed['pred']) if _pred_objects(dict(parsed,pred=[obj]))[0]]
        record['projected_to_parsed_prediction_index']=projection
        prefix_text=tokenizer.decode(job['prefix'],skip_special_tokens=False)
        prefix_parsed=native_record(prefix_text,{'row_id':case['example_id']},case['golden'],'conditional')
        release_rows=len(prefix_parsed['pred'])+1
        direct=incidence(parsed);free=incidence(parsed,release_rows)
        before=score(prefix_parsed,seed=-1,length=len(job['prefix']),stop='conditional')['50']['owners']
        release_ids=job['prefix']+case['A' if record['arm']=='partial_A' else
                    {'B':'B','A':'A','Aprime':'Aprime','sample_A':'A'}[record['arm']]]['ids']
        # Partial release is inside a row, so its completed row is observed, not forced A.
        if record['arm']=='partial_A':
            close=tokenizer.convert_tokens_to_ids('<|box_end|>')
            end=next((i+1 for i in range(len(job['prefix']),len(ids)) if ids[i]==close),len(ids))
            release_ids=ids[:end]
        released=native_record(tokenizer.decode(release_ids,skip_special_tokens=False),{'row_id':case['example_id']},case['golden'],'conditional')
        covered=score(released,seed=-1,length=len(release_ids),stop='conditional')['50']['owners']
        all_owners={str(g['object_id']) for g in case['golden']['gt']}
        record['conditional']=dict(prefix_covered=before,release_covered=covered,
            remaining_obligations=sorted(all_owners-set(covered)),
            target_A_direct=bool(direct[case['owner']]),target_A_globally_assigned=case['owner'] in record['score']['50']['owners'],
            B_free_suffix_direct=bool(free[case['B_owner']]),
            B_free_suffix_globally_assigned=any(m['owner']==case['B_owner'] and projection[m['pred_index']]>=release_rows for m in record['score']['50']['matches']),
            direct_action_incidence=direct,free_suffix_incidence=free,
            direct_incidence_by_threshold={str(t):incidence(parsed,threshold=t/100) for t in (50,60,80)},
            any_category_incidence=incidence(parsed,category_consistent=False),
            direct_but_unassigned=sorted(o for o,js in direct.items() if js and o not in record['score']['50']['owners']),
            free_row_count=max(0,len(parsed['pred'])-release_rows))
    require(seen==set(expected),'missing output IDs')
    return records


def reduce_records(records, manifest):
    cases=indexed(manifest['selected'],'case_id');out=[]
    for cid in sorted({r['case_id'] for r in records}):
        arms={r['arm']:r for r in records if r['case_id']==cid};base=arms['B'];case=cases[cid]
        source_suffix=set(m['owner'] for m in base['score']['50']['matches'] if m['pred_index']>case['boundary'])
        comparisons={}
        for arm,row in arms.items():
            comparisons[arm]={}
            for threshold in ('50','60','80'):
                old=set(base['score'][threshold]['owners']);new=set(row['score'][threshold]['owners'])
                comparisons[arm][threshold]=dict(gained=sorted(new-old),lost=sorted(old-new),
                    matching_reassignment_gains=sorted(o for o in new-old if base['conditional']['direct_incidence_by_threshold'][threshold][o]),
                    matching_reassignment_losses=sorted(o for o in old-new if row['conditional']['direct_incidence_by_threshold'][threshold][o]),
                    category_correction_possible_gains=sorted(o for o in new-old if not base['conditional']['direct_action_incidence'][o]
                        and base['conditional']['any_category_incidence'][o]),
                    iou_threshold_possible_gains=sorted(o for o in new-old if threshold!='50' and
                        base['conditional']['direct_action_incidence'][o] and not base['conditional']['direct_incidence_by_threshold'][threshold][o]))
            new=set(row['score']['50']['owners'])
            comparisons[arm]['old_suffix_owners_lost']=sorted(source_suffix-new)
            comparisons[arm]['old_suffix_owners_retained']=sorted(source_suffix&new)
        common=set(arms['A']['conditional']['remaining_obligations']) & set(arms['sample_A']['conditional']['remaining_obligations'])
        residual_a=set(arms['A']['conditional']['remaining_obligations'])
        residual_sample=set(arms['sample_A']['conditional']['remaining_obligations'])
        workload_compatible=residual_a==residual_sample
        horizon=min(arms[a]['conditional']['free_row_count'] for a in arms)
        # Compare direct incidence in an identical free-row horizon; retain global full-output sets above.
        progress={}
        for arm,row in arms.items():
            release=len(row['parsed']['pred'])-row['conditional']['free_row_count']
            progress[arm]=dict(common_remainder_recovered=sorted(o for o in common if row['conditional']['free_suffix_incidence'][o]),
                common_horizon_direct_owners=sorted(o for o,js in row['conditional']['free_suffix_incidence'].items() if any(j<release+horizon for j in js)),
                common_remainder_horizon_recovered=sorted(o for o in common if any(j<release+horizon for j in row['conditional']['free_suffix_incidence'][o])))
        out.append(dict(case_id=cid,comparisons=comparisons,common_remaining_obligations=sorted(common),
            release_workloads_identical=workload_compatible,
            natural_only_remaining_obligations=sorted(residual_a-residual_sample),
            sampled_only_remaining_obligations=sorted(residual_sample-residual_a),
            sampled_history_control_discriminating=bool(common) and workload_compatible,
            sampled_history_control_limitation=None if common and workload_compatible else
                ('empty common remainder' if not common else 'incompatible complete workload; shared obligations descriptive only'),
            common_free_row_horizon=horizon,progress=progress))
    return out


def execute(output, manifest_path, admission_path):
    import torch
    import yaml
    from src.qwen.native import prepare_native_inputs
    from src.qwen.generation import generate_continuations
    manifest=json.loads(manifest_path.read_text());admission=json.loads(admission_path.read_text())
    require(admission.get('manifest_file_sha256',file_hash(manifest_path))==file_hash(manifest_path),'admission manifest byte identity')
    cases=validate_admission(manifest,admission)
    require(cases,'empty admitted panel; no model needed')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='1','GPU1 only')
    require(torch.cuda.is_available() and torch.cuda.device_count()==1,'exactly one visible GPU')
    run_dir=output/'execution';run_dir.mkdir(exist_ok=False)
    started=time.monotonic();receipt=dict(status='running',model_loads=0,model_forwards=0,image_forwards=0,
        continuations=0,new_tokens=0,manifest_sha256=digest(manifest),admission_sha256=file_hash(admission_path),pid=os.getpid())
    publish(run_dir/'launch.json',receipt)
    def timeout(*_): raise TimeoutError('3600 second cumulative model execution budget')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(3600)
    try:
        for path,sha in manifest['source_files'].items(): require(file_hash(path)==sha,'frozen source changed')
        config=yaml.safe_load((SOURCE_ROOT/'configs/resolved.yaml').read_text())['config']
        require(config['model']['dtype']=='fp32' and config['backend']['hf']==dict(attn_implementation='sdpa',patch_embed_linearization='enabled'),'frozen numerics')
        require(config['generation']['max_new_tokens']==CAP and config['generation']['temperature']==0
                and config['generation']['top_p']==1 and config['generation']['repetition_penalty']==1,'frozen decode')
        source=json.loads((SOURCE_ROOT/'run_manifest.json').read_text())
        plan=json.loads((UPDATE_ROOT/'round-1/plan.json').read_text())
        require(config['model']['base_model']==plan['model']['base_model_path'] and
                config['adapter']['path']==plan['model']['current_adapter']['root'] and
                config['embedding_delta']['path']==plan['model']['source_embedding']['root'],'Source config/retained plan identity')
        require(source['model_identity']['adapter']['enabled'] is True and
                not source['model_identity']['adapter']['merged_adapters'],'unmerged Source required')
        binding=dict(model_identity=source['model_identity'],tokenizer_identity=source['tokenizer_identity'],
                     runtime_effective_settings=source['backend_session']['effective_settings'])
        publish(run_dir/'effective-config.json',dict(config=config,source=binding,device='cuda:0',visible_device='1'))
        qwen,receipt['loaded_identity']=load_components(config,binding,torch.device('cuda:0'));receipt['model_loads']=1
        train=indexed(rows(config['data']['input_jsonl']),'image_id');images=indexed(rows(SOURCE_ROOT/'image_plan.jsonl'),'row_id')
        request_cases=[]
        for c in cases:
            g=c['golden']; request_cases.append(dict(row_id=c['example_id'],row_index=g['row_index'],
                image_path=g['image_path'],image_width=g['image_width'],image_height=g['image_height'],
                input_record=train[str(c['image_id'])],image_plan=images[c['example_id']]))
        requests,prompt_metadata=build_requests(qwen,config,request_cases)
        publish(run_dir/'prompt-metadata.json',prompt_metadata)
        def count_model(*_): receipt['model_forwards']+=1
        def count_image(*_): receipt['image_forwards']+=1
        qwen.model.register_forward_pre_hook(count_model)
        visual=[m for n,m in qwen.model.named_modules() if n.endswith('visual')]
        require(len(visual)==1,'visual counter ambiguity');visual[0].register_forward_pre_hook(count_image)
        torch.cuda.reset_peak_memory_stats()
        expected=[];row_path=run_dir/'rows.jsonl'
        with row_path.open('x') as stream:
            for c,request in zip(cases,requests,strict=True):
                require(list(request.expected_token_ids)==c['group']['prompt_token_ids'],'bank/native exact prompt tokens')
                batch=prepare_native_inputs(qwen.processor,[request],device=torch.device('cuda:0'),record_media_identity=True)
                require(batch.media_sha256[0]==c['group']['executed_media_sha256'],'media bytes identity')
                for arm in ('B','partial_A','A','Aprime','sample_A'):
                    job=branch(c,arm);tick=time.monotonic()
                    require(receipt['continuations']<60 and receipt['new_tokens']+job['remaining']<=191208,'execution count/token budget')
                    result=generate_continuations(qwen.model,batch,extensions=[job['prefix']+job['forced']],budgets=[job['remaining']],
                        eos_token_id=151645,pad_token_id=qwen.tokenizer.pad_token_id,trace='none')[0]
                    require(result.request_id==c['example_id'],'native continuation request association')
                    torch.cuda.synchronize();receipt['continuations']+=1;receipt['new_tokens']+=len(result.token_ids)
                    ids=job['prefix']+job['forced']+list(result.token_ids);text=qwen.tokenizer.decode(ids,skip_special_tokens=False)
                    row=dict(case_id=c['case_id'],arm=arm,manifest_sha256=digest(manifest),action_ids=ids,
                        prefix_ids=job['prefix'],forced_ids=job['forced'],suffix_ids=list(result.token_ids),remaining_budget=job['remaining'],
                        text=text,stop_reason=result.stop_reason,seconds=time.monotonic()-tick,
                        parsed=native_record(text,{'row_id':c['example_id']},c['golden'],result.stop_reason))
                    stream.write(json.dumps(row)+'\n');stream.flush();os.fsync(stream.fileno())
                    expected.append((c['case_id'],arm))
                    cold=consume(row_path,manifest,qwen.tokenizer,expected)
                    if receipt['continuations']==1:
                        require(ids==c['source_ids'],'first B retained token diagonal differs')
                        require(row['parsed']==c['golden'],'first B retained parser diagonal differs')
                        golden_score=score(c['golden'],seed=-1,length=len(ids),stop=result.stop_reason)
                        require(cold[0]['score']==golden_score,'first B owner consumer diagonal differs')
                        publish(run_dir/'first-B-diagonal.json',dict(status='verified',case_id=c['case_id'],tokens=len(ids),score=golden_score))
        cold=consume(row_path,manifest,qwen.tokenizer,expected)
        publish(run_dir/'consumer.json',cold);publish(run_dir/'reduction.json',reduce_records(cold,manifest))
        receipt['status']='completed'
    except BaseException as exc:
        receipt.update(status='failed',error=repr(exc),traceback=traceback.format_exc())
        raise
    finally:
        signal.alarm(0);receipt.update(elapsed_seconds=time.monotonic()-started,
            rss_peak_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0,
            artifact_bytes=sum(p.stat().st_size for p in run_dir.rglob('*') if p.is_file()))
        publish(run_dir/'terminal.json',receipt)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--execute',action='store_true');parser.add_argument('--admission',type=Path)
    parser.add_argument('--manifest',type=Path)
    args=parser.parse_args()
    if args.execute:
        require(args.admission is not None,'explicit root admission required')
        execute(args.output,args.manifest or args.output/'manifest.json',args.admission)
    else: census(args.output)


if __name__=='__main__': main()
