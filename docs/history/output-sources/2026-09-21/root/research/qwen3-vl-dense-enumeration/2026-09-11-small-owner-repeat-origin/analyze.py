"""Cold native readback and descriptive prefix/image contrasts, no model calls."""
from pathlib import Path
import argparse
import importlib.util
import json
import statistics
import sys

ROOT = Path(__file__).parent
sys.path.insert(0, '/data/CoordExp/.worktrees/research-probes')
from tokenizers import Tokenizer
from probes.dora_owner_learning.geometric_dedup import trajectory_layout, _exact_token_text_frame
from probes.source_rweak_row_cross.run import native_record
from probes.dora_owner_learning.candidate_opportunity import file_hash


def module(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value


census = module('census_builder', ROOT/'census/build_census.py')
runner = module('native_probe', ROOT/'run_probe.py')


def parse_rows(ids, case, tok):
    text, spans=_exact_token_text_frame(ids,tok)
    width,height=case['source_case']['image_width'],case['source_case']['image_height']
    layout=trajectory_layout(ids,tok,image_width=width,image_height=height,row_id=case['case_id'])
    rows=census.build_raw_rows(case=dict(action_ids=ids,image_width=width,image_height=height),
        layout=layout,action_text=text,token_spans=spans,
        token_pieces=[tok.decode([i],skip_special_tokens=False) for i in ids])
    assert sum(r['strict_iou_gt_0_95_repeat'] for r in rows)==len(layout['duplicate_row_indices'])
    return rows, layout


def center(box):
    return [(box[0]+box[2])/2,(box[1]+box[3])/2]


def geometry_readout(rows, seed):
    first=rows[:20]
    a,b=center(seed['original_bins']),center(seed['translated_bins'])
    shift=[b[i]-a[i] for i in (0,1)];norm=sum(x*x for x in shift)
    assert norm>0
    values=[]
    for row in first:
        c=center(row['coord_bins'])
        projection=sum((c[i]-a[i])*shift[i] for i in (0,1))/norm
        values.append(dict(index=row['raw_row_index'],description=row['description'],
            bins=row['coord_bins'],valid=row['geometry_valid'],center=c,
            shift_projection=projection,seed_class=row['description']==seed['description']))
    same=[v['shift_projection'] for v in values[:5] if v['seed_class']]
    return dict(first20=values,first5_count=min(5,len(values)),
        first5_seed_class_count=len(same),first20_seed_class_count=sum(x['seed_class'] for x in values),
        median_projection_first5_seed_class=statistics.median(same) if same else None,
        projection_meaning='0 is old seed center along shift;1 is translated seed center; orthogonal distance is not represented')


def summarize(record, case, tok):
    ids=record['action_ids'];prefix=record['extension_ids'];free=record['free_ids']
    assert ids==prefix+free and tok.decode(ids,skip_special_tokens=False)==record['text']
    cold=native_record(record['text'],{'row_id':case['source_case']['row_id']},case['golden'],record['stop_reason'])
    assert cold==record['parsed'], 'native parsed readback changed'
    rows,layout=parse_rows(ids,case,tok)
    prefix_chars=len(tok.decode(prefix,skip_special_tokens=False))
    free_rows=[r for r in rows if r['char_start']>=prefix_chars]
    complete=[r for r in free_rows if r['complete_canonical_row']]
    seen=set();exact=0
    for r in rows:
        if not r['complete_canonical_row']:continue
        key=(r['description'],tuple(r['coord_bins']))
        if r['char_start']>=prefix_chars and key in seen:exact+=1
        seen.add(key)
    result=dict(case_id=case['case_id'],job_id=record['job_id'],
        boundary=record['boundary'],image_condition=record['image_condition'],history_condition=record['history_condition'],
        free_tokens=len(free),prefix_tokens=len(prefix),stop=record['stop_reason'],
        free_raw_starts=len(free_rows),free_complete_rows=len(complete),
        free_valid_rows=sum(r['geometry_valid'] for r in free_rows),
        free_geometry_invalid_rows=sum(r['native_status']=='geometry_invalid' for r in free_rows),
        free_other_malformed_rows=sum(not r['geometry_valid'] and r['native_status']!='geometry_invalid' for r in free_rows),
        free_strict_repeats_against_all_earlier=sum(r['strict_iou_gt_0_95_repeat'] for r in free_rows),
        free_exact_description_geometry_repeats_including_invalid=exact,
        complete_description_counts={d:sum(r['description']==d for r in complete) for d in sorted({r['description'] for r in complete})},
        first5_rows=[{k:r[k] for k in ('description','coord_bins','geometry_valid','raw_row_index')} for r in complete[:5]])
    assert result['free_valid_rows']+result['free_geometry_invalid_rows']+result['free_other_malformed_rows']==len(free_rows)
    if record['boundary'] in ('early','late'):
        result['geometry']=geometry_readout(complete,case['seeds'][record['boundary']])
        expected=case['baseline_action_ids'][len(prefix):len(prefix)+len(free)]
        result['saved_native_suffix_exact_match']=free==expected if record['image_condition']=='original' and record['history_condition']=='native' else None
    return result


def main(smoke):
    packet=json.loads((ROOT/'packet.json').read_text());ph=file_hash(ROOT/'packet.json')
    plan=runner.validate_packet(packet);tok=Tokenizer.from_file(packet['model']['base_model_path']+'/tokenizer.json')
    roots=[ROOT/'smoke-01'] if smoke else [ROOT/'full'/f'rank-{i}' for i in range(8)]
    results=[];terminals=[];bindings={str(ROOT/'packet.json'):ph,str(Path(__file__)):file_hash(__file__)}
    for root in roots:
        terminal=json.loads((root/'terminal.json').read_text());assert terminal['status']=='completed'
        rank=terminal['rank'];case=packet['cases'][rank];records=runner.read_jsonl(root/'records.jsonl')
        runner.validate_readback(records,cases=[plan['cases'][rank]],packet_sha256=ph,smoke=smoke)
        assert len(records)==terminal['continuations']
        assert sum(len(r['free_ids']) for r in records)==terminal['new_tokens']
        results.extend(summarize(r,case,tok) for r in records);terminals.append(terminal)
        for name in ('terminal.json','records.jsonl','batches.json','model.json'):
            bindings[str(root/name)]=file_hash(root/name)
    contrasts=[];byid={(r['case_id'],r['job_id']):r for r in results}
    for case in packet['cases']:
        cid=case['case_id']
        for boundary in ('early','late'):
            for image in ('original','donor'):
                a=byid.get((cid,f'{boundary}_{image}_native'));b=byid.get((cid,f'{boundary}_{image}_translated'))
                if a is None or b is None:continue
                x=a['geometry']['median_projection_first5_seed_class'];y=b['geometry']['median_projection_first5_seed_class']
                contrasts.append(dict(case_id=cid,boundary=boundary,image_condition=image,
                    native_stop=a['stop'],translated_stop=b['stop'],
                    native_free_rows=a['free_complete_rows'],translated_free_rows=b['free_complete_rows'],
                    native_seed_class_first5=a['geometry']['first5_seed_class_count'],
                    translated_seed_class_first5=b['geometry']['first5_seed_class_count'],
                    median_center_shift_response=None if x is None or y is None else y-x))
    donors=[]
    if not smoke:
        seen=set()
        for case in packet['cases']:
            if case['donor_image_id'] in seen:continue
            seen.add(case['donor_image_id']);ids=case['donor_baseline_action_ids']
            if ids is None:
                row=byid[(case['case_id'],'donor_natural')]
                donors.append(dict(image_id=case['donor_image_id'],source='fresh_native_donor',
                    counts=row['complete_description_counts'],stop=row['stop'],tokens=row['free_tokens']))
            else:
                rs,_=parse_rows(ids,case,tok)
                donors.append(dict(image_id=case['donor_image_id'],source='hash_bound_existing_Stable50',
                    counts={d:sum(r['description']==d for r in rs) for d in sorted({r['description'] for r in rs})},
                    stop='im_end' if ids[-1]==151645 else 'length',tokens=len(ids)))
    result=dict(schema='small_owner_repeat_origin.reduction.v1',source_files=bindings,
        scope='smoke' if smoke else 'full_panel',cells=len(results),results=results,
        coordinate_contrasts=contrasts,donor_alone_controls=donors,
        resources=dict(continuations=sum(t['continuations'] for t in terminals),
            free_tokens=sum(t['new_tokens'] for t in terminals),
            model_loads=sum(t['model_loads'] for t in terminals),
            model_forwards=sum(t['model_forwards'] for t in terminals),
            image_forwards=sum(t['image_forwards'] for t in terminals),
            allocated_gpu_seconds=sum(t['elapsed_seconds'] for t in terminals)),
        interpretation='Conditional-only diagnostics; no owner recovery from artificial prefixes, no claims of prevalence or internal circuit identity')
    output=ROOT/('smoke-reduction.json' if smoke else 'reduction.json')
    output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'cells':len(results),'resources':result['resources'],'donor_controls':donors,'coordinate_contrasts':contrasts},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');main(parser.parse_args().smoke)
