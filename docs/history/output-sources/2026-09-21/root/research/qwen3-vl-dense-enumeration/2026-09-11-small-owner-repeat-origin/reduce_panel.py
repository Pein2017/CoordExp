"""Merge six preserved ranks and two envelope-repaired ranks without replaying them."""
from pathlib import Path
import argparse
import importlib.util
import json

ROOT=Path(__file__).parent
spec=importlib.util.spec_from_file_location('analysis_core',ROOT/'analyze.py')
core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)


def main(partial):
    paths=[ROOT/'packet.json',ROOT/'packet-repair-01.json']
    packets=[json.loads(p.read_text()) for p in paths]
    plans=[core.runner.validate_packet(p) for p in packets]
    old,repaired=packets
    assert old['config']==repaired['config'] and old['model']==repaired['model']
    for a,b in zip(old['cases'],repaired['cases']):
        for key in ('case_id','source_case','baseline_action_ids','prompt_token_ids','golden','jobs','seeds'):
            assert a[key]==b[key], f'scientific input changed: {key}'
        for key in ('image_path','image_width','image_height','image_plan'):
            assert a['donor_case'][key]==b['donor_case'][key]
    tok=core.Tokenizer.from_file(old['model']['base_model_path']+'/tokenizer.json')
    ranks=[0,1,2,4,5,6] if partial else list(range(8))
    results=[];terminals=[];bindings={str(p):core.file_hash(p) for p in paths+[Path(__file__),ROOT/'analyze.py']}
    for rank in ranks:
        retry=rank in (3,7);packet=packets[int(retry)];plan=plans[int(retry)]
        root=ROOT/('repair-01' if retry else 'full')/f'rank-{rank}'
        terminal=json.loads((root/'terminal.json').read_text());assert terminal['status']=='completed'
        ph=core.file_hash(paths[int(retry)]);assert terminal['packet_sha256']==ph
        raw=core.runner.read_jsonl(root/'records.jsonl')
        core.runner.validate_readback(raw,cases=[plan['cases'][rank]],packet_sha256=ph)
        assert len(raw)==terminal['continuations'] and sum(len(r['free_ids']) for r in raw)==terminal['new_tokens']
        results.extend(core.summarize(r,packet['cases'][rank],tok) for r in raw);terminals.append(terminal)
        for name in ('terminal.json','records.jsonl','batches.json','model.json'):
            bindings[str(root/name)]=core.file_hash(root/name)
    byid={(r['case_id'],r['job_id']):r for r in results};coordinate=[];image=[]
    for rank in ranks:
        case=old['cases'][rank];cid=case['case_id']
        for boundary in ('early','late'):
            for im in ('original','donor'):
                a,b=[byid[(cid,f'{boundary}_{im}_{h}')] for h in ('native','translated')]
                x=a['geometry']['median_projection_first5_seed_class'];y=b['geometry']['median_projection_first5_seed_class']
                coordinate.append(dict(case_id=cid,boundary=boundary,image_condition=im,
                    native_stop=a['stop'],translated_stop=b['stop'],
                    native_free_rows=a['free_complete_rows'],translated_free_rows=b['free_complete_rows'],
                    native_seed_class_first5=a['geometry']['first5_seed_class_count'],
                    translated_seed_class_first5=b['geometry']['first5_seed_class_count'],
                    median_center_shift_response=None if x is None or y is None else y-x))
            for h in ('native','translated'):
                a,b=[byid[(cid,f'{boundary}_{im}_{h}')] for im in ('original','donor')]
                image.append(dict(case_id=cid,boundary=boundary,history_condition=h,
                    original_stop=a['stop'],donor_stop=b['stop'],
                    original_seed_class_first20=a['geometry']['first20_seed_class_count'],
                    donor_seed_class_first20=b['geometry']['first20_seed_class_count'],
                    original_free_tokens=a['free_tokens'],donor_free_tokens=b['free_tokens']))
    donors=[];seen=set()
    for rank in ranks:
        case=old['cases'][rank];did=case['donor_image_id']
        if did in seen:continue
        seen.add(did);ids=case['donor_baseline_action_ids']
        if ids is None:
            row=byid[(case['case_id'],'donor_natural')]
            donors.append(dict(image_id=did,source='fresh_native_donor',counts=row['complete_description_counts'],stop=row['stop'],tokens=row['free_tokens']))
        else:
            rows,_=core.parse_rows(ids,case,tok)
            donors.append(dict(image_id=did,source='hash_bound_existing_Stable50',
                counts={d:sum(r['description']==d for r in rows) for d in sorted({r['description'] for r in rows})},
                stop='im_end' if ids[-1]==151645 else 'length',tokens=len(ids)))
    smoke=json.loads((ROOT/'smoke-01/terminal.json').read_text())
    failures=[json.loads((ROOT/f'full/rank-{i}/terminal.json').read_text()) for i in (3,7)]
    assert all(t['continuations']==t['model_forwards']==t['new_tokens']==0 for t in failures)
    resources=dict(continuations=sum(t['continuations'] for t in terminals),
        free_tokens=sum(t['new_tokens'] for t in terminals),model_loads=sum(t['model_loads'] for t in terminals),
        model_forwards=sum(t['model_forwards'] for t in terminals),image_forwards=sum(t['image_forwards'] for t in terminals),
        allocated_gpu_seconds=sum(t['elapsed_seconds'] for t in terminals),
        smoke_continuations=smoke['continuations'],smoke_free_tokens=smoke['new_tokens'],smoke_model_loads=smoke['model_loads'],smoke_allocated_gpu_seconds=smoke['elapsed_seconds'],
        failed_model_loads=sum(t['model_loads'] for t in failures),failed_allocated_gpu_seconds=sum(t['elapsed_seconds'] for t in failures),
        peak_cuda_allocated_bytes=max(t['peak_cuda_allocated_bytes'] for t in terminals+[smoke]),
        peak_rss_bytes=max(t['host_peak_rss_bytes'] for t in terminals+[smoke]))
    assert partial or resources['continuations']==74
    result=dict(schema='small_owner_repeat_origin.reduction.v1',scope='partial_six' if partial else 'full_panel',
        source_files=bindings,cells=len(results),case_count=len(ranks),results=results,
        coordinate_contrasts=coordinate,image_contrasts=image,donor_alone_controls=donors,resources=resources,
        execution_repair='Only ranks3/7 donor input-envelope schema; six completed ranks preserved; original failure receipts retained; same inference inputs verified on CPU',
        interpretation='Conditional dependence only; no ownership credit from artificial prefixes, no population prevalence, no circuit or training-origin claim')
    output=ROOT/('partial-six.json' if partial else 'reduction.json')
    with output.open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps(dict(cells=len(results),resources=resources,donors=donors),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--partial',action='store_true');main(parser.parse_args().partial)
