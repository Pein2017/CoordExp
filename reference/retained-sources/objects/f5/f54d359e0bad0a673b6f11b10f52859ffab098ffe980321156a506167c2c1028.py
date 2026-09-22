"""Deterministic finite review remainder. No image viewing, labels or inference."""
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
from probes.owner_successor_scale import data as d

BASE=d.ROOT.parent
OUT=BASE/'physical-admission-remainder-inventory'


def main():
    recovery=BASE/'supply-recovery'
    completion=d.e.read(recovery/'completion.json')
    pool=d.e.read(BASE/'supply/pool.json')
    jobs=d.e.read(recovery/'ordered-jobs.json')['jobs']
    results={r['job_id']:r for r in d.e.read(recovery/'ordered-results.json')['rows']}
    images={r['image_id']:r for r in d.e.read(recovery/'image-records.json')['records']}
    jobmap={j['job_id']:j for j in jobs}
    candidates={j['job_id'] for j in jobs if results[j['job_id']]['local_w']['status']=='candidate_local_w'}
    assert len(jobs)==len(results)==589 and len(candidates)==502
    def visual_key(jid):
        j=jobmap[jid];w=results[jid]['local_w']
        return (j['image_id'],j['c_description'],tuple(j['c_bbox']),w['w_description'],tuple(w['w_bbox_xyxy_pixels']))
    prior_bindings={};indexed=set();prior_groups=defaultdict(list);prior_counts={}
    for name in ('physical-admission','physical-admission-preserved','physical-admission-recovery'):
        path=BASE/name/'review-index-v1.json';index=d.e.read(path)
        prior_bindings[name]=d.e.binding(path)
        ids=[r['job_id'] for r in index['rows']]
        assert len(ids)==len(set(ids)) and not indexed.intersection(ids)
        assert set(ids)<=candidates
        indexed.update(ids)
        for group in index['groups']:
            keys={visual_key(jid) for jid in group['job_ids']}
            assert len(keys)==1,'existing visual group differs from exact c/w key'
            prior_groups[next(iter(keys))].append(group['visual_group_id'])
        prior_counts[name]={'candidate_rows':len(ids),'named_groups':len(index['groups']),'images':len({jobmap[jid]['image_id'] for jid in ids})}
    remaining=candidates-indexed
    # Source identities include execution stage, so shard0 from two invocations
    # cannot alias. Retain exact raw-line SHA and1-based position per history.
    source_rows={};source_jobs=defaultdict(list);source_images=defaultdict(list);source_bindings={}
    for entry in completion['source_bindings']:
        binding=entry['binding'];path=Path(binding['path'])
        assert d.e.binding(path)==binding
        rel=path.relative_to(BASE)
        sid='-'.join(rel.parts[:-1])
        source_bindings.setdefault(sid,{})[path.stem]=binding
        for line_number,line in enumerate(path.read_text().splitlines(),1):
            row=json.loads(line);jid=row.get('job_id')
            if path.stem!='images' and jid not in remaining:continue
            ident={'binding':binding,'line_1based':line_number,'line_sha256':hashlib.sha256(line.encode()).hexdigest(),'source_shard_id':sid}
            if path.stem=='rows':
                assert jid not in source_rows
                source_rows[jid]=(row,ident)
            elif path.stem=='jobs':source_jobs[jid].append((row,ident))
            else:source_images[row['image_id']].append(ident)
    assert set(source_rows)==remaining
    cards={c['job_id']:c['card'] for c in completion['cards']}
    cards.update({c['job_id']:c['card'] for c in d.e.read(recovery/'preserved-cards-manifest.json')['cards']})
    groups=[];rows=[];group_keys={}
    for job in jobs:  # Already verified canonical frozen pool/history/c order.
        jid=job['job_id']
        if jid not in remaining:continue
        result,ri=source_rows[jid];assert result==results[jid]
        exact_job,ji=next((r,i) for r,i in source_jobs[jid] if i['source_shard_id']==ri['source_shard_id'])
        assert exact_job==job
        key=visual_key(jid);card=cards[jid];assert d.e.binding(card['path'])==card
        if key not in group_keys:
            gid=f'PAM-{len(groups)+1:04d}';group_keys[key]=gid
            groups.append({'visual_group_id':gid,'visual_group_rank':len(groups)+1,'image_id':job['image_id'],'job_ids':[],'representative_card':{'path':card['path'],'binding':card,'representative_job_id':jid},'prior_exact_visual_alias_group_ids':prior_groups.get(key,[]),'exact_history_newness_reviewed_per_job':True,'review_status':'pending_luna_individual_review','training_target':False})
        gid=group_keys[key];group=groups[int(gid.split('-')[1])-1];group['job_ids'].append(jid)
        case=images[job['image_id']]['frozen']['case'];w=result['local_w']
        def axis(desc,box,text,tokens):
            return {'description':desc,'bbox_native_pixels_xyxy':box,'literal_text':text,'token_ids_sha256':hashlib.sha256(json.dumps(tokens,separators=(',',':')).encode()).hexdigest(),'physical_review':{'status':'pending_luna_individual_review'}}
        rows.append({'job_id':jid,'image_id':job['image_id'],'example_id':job['example_id'],'visual_group_id':gid,'shard':ri['source_shard_id'],'source_identity':{'candidate_index':job['candidate_index'],'history_index':job['history_index'],'later_row_ordinal':job['later_row_ordinal'],'row_id':jid,'output_shard':ri['source_shard_id'],'output_status':w['status'],'packet':result['packet']},'source_job':ji,'source_row':ri,'source_image_records':source_images[job['image_id']],'source_job_ordinal':ji['line_1based']-1,'source_row_ordinal':ri['line_1based']-1,'source_job_sha256':ji['line_sha256'],'source_row_sha256':ri['line_sha256'],'image':{'path':case['image_path'],'sha256':case['image_plan']['image_content_sha256'],'dimensions_native_pixels':[case['image_width'],case['image_height']],'image_id':job['image_id'],'example_id':job['example_id']},'c':axis(job['c_description'],job['c_bbox'],job['c_text'],job['c_token_ids']),'w':axis(w['w_description'],w['w_bbox_xyxy_pixels'],w['w_text'],w['w_token_ids']),'exact_history':{'literal_text':job['h_text'],'token_ids_sha256':hashlib.sha256(json.dumps(job['h_token_ids'],separators=(',',':')).encode()).hexdigest(),'history_boundary':job['history_boundary']},'first_owner_axis':job['first_owner'],'repeat_bbox':job['repeat_bbox'],'review_evidence':{'card_path':card['path'],'card_binding':card},'execution_history_preserved':True,'training_target':False})
    assert {r['job_id'] for r in rows}==remaining and indexed|remaining==candidates
    assert not indexed&remaining
    prior_gids={g for values in prior_groups.values() for g in values}
    assert not prior_gids&{g['visual_group_id'] for g in groups}
    counts={'closed_acquisition_images':4096,'closed_conditional_outcomes':589,'all_machine_candidate_rows':502,'already_indexed_rows':len(indexed),'remaining_candidate_rows':len(rows),'already_indexed_exact_visual_groups':len(prior_groups),'remaining_exact_visual_groups':len(groups),'already_indexed_images':len({jobmap[j]['image_id'] for j in indexed}),'remaining_images':len({r['image_id'] for r in rows}),'remaining_source_shards':len({r['shard'] for r in rows}),'remaining_rows_by_source_shard':dict(Counter(r['shard'] for r in rows)),'remaining_groups_with_prior_exact_visual_alias':sum(bool(g['prior_exact_visual_alias_group_ids']) for g in groups),'prior_indexes':prior_counts}
    common={'schema':'owner_successor_scale.physical_review_index.v1','status':'frozen_inventory_only_no_review','scope':'Complete unindexed candidate_local_w remainder of the same closed4096-image/589-outcome acquisition; canonical pool/history/c order, exact c/w visual key.','admission_boundary':'Inventory only. No labels, no GPU/view_image calls, no root admission or training export. Review each exact history; no-w/noncandidate rows are not substituted.','review_route':'Root assigns Luna reviewers; individual existing full cards, no shrinking/montage.','source_bindings':{'completion':d.e.binding(recovery/'completion.json'),'pool':d.e.binding(BASE/'supply/pool.json'),'canonical_jobs':d.e.binding(recovery/'ordered-jobs.json'),'canonical_results':d.e.binding(recovery/'ordered-results.json'),'canonical_images':d.e.binding(recovery/'image-records.json'),'excluded_review_indexes':prior_bindings,'source_shards':source_bindings},'counts':counts}
    OUT.mkdir(exist_ok=True)
    index_path=OUT/'review-index-v1.json';d.e.publish(index_path,{**common,'groups':groups,'rows':rows})
    batches=[]
    for start in range(0,len(groups),40):
        subset=groups[start:start+40];gids={g['visual_group_id'] for g in subset};selected=[r for r in rows if r['visual_group_id'] in gids]
        number=len(batches)+1;path=OUT/f'batch-{number:02d}-review-index.json'
        d.e.publish(path,{**common,'parent_index':d.e.binding(index_path),'batch_number':number,'groups':subset,'rows':selected})
        batches.append({'batch_id':f'PAM-B{number:02d}','index':d.e.binding(path),'group_ids':[g['visual_group_id'] for g in subset],'job_ids':[r['job_id'] for r in selected],'group_count':len(subset),'row_count':len(selected),'image_ids':list(dict.fromkeys(g['image_id'] for g in subset)),'assignment_status':'unassigned_root_dispatch_only'})
    assert [gid for b in batches for gid in b['group_ids']]==[g['visual_group_id'] for g in groups]
    assert {jid for b in batches for jid in b['job_ids']}==remaining
    d.e.publish(OUT/'assignment-manifest-v1.json',{'schema':'owner_successor_scale.finite_remainder_review_assignment.v1','status':'frozen_unassigned','index':d.e.binding(index_path),'counts':counts,'batches':batches,'coverage':{'candidate_union_complete':True,'indexed_remaining_disjoint':True,'each_remaining_job_once':True,'max40_consecutive_groups_per_batch':True},'script':d.e.binding(Path(__file__))})
    print(counts);print('batches',[(b['batch_id'],b['group_count'],b['row_count']) for b in batches]);print(d.e.binding(OUT/'assignment-manifest-v1.json'))


if __name__=='__main__':main()
