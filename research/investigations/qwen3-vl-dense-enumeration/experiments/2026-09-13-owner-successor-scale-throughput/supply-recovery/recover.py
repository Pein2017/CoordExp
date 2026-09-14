"""One-off exact unfinished-call continuation; original producer stays immutable."""
import argparse
from collections import Counter
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

from probes.owner_successor_scale import data as d

ROOT=d.ROOT.parent/'supply-recovery'
SOURCE=d.ROOT/'remainder-v1'
OLD_PACKET=d.ROOT/'remainder-packet-v1.json'
SCRIPT=Path(__file__).resolve()


def complete_lines(path):
    data=Path(path).read_bytes();lines=data.splitlines(keepends=True);rows=[]
    rejected=[]
    for index,line in enumerate(lines):
        try:
            assert line.endswith(b'\n')
            rows.append(json.loads(line))
        except (ValueError,AssertionError):
            assert index==len(lines)-1,'nonterminal corrupt source line'
            rejected.append({'line':index+1,'bytes':len(line)})
    return rows,rejected


def snapshot():
    from transformers import AutoTokenizer
    packet=d.e.read(OLD_PACKET);pool=d.e.read(packet['pool']['path'])
    assert d.e.binding(Path(d.__file__))==packet['producer']
    tokenizer=AutoTokenizer.from_pretrained(pool['config']['model']['base_model'],local_files_only=True)
    records,jobs,rows,sources,terminals=[],[],[],[],[]
    for shard in range(4):
        root=SOURCE/f'shard-{shard}'
        terminal=d.e.read(root/'terminal.json');terminals.append(terminal)
        assert terminal['status'] in ('failed','completed')
        assert terminal['packet']==d.e.binding(OLD_PACKET)
        for name,destination in [('images',records),('jobs',jobs),('rows',rows)]:
            path=root/f'{name}.jsonl';values,rejected=complete_lines(path)
            sources.append({'binding':d.e.binding(path),'rejected_partial_tail':rejected})
            destination.extend(values)
    by_image={r['image_id']:r for r in records};by_job={j['job_id']:j for j in jobs}
    assert len(by_image)==len(records) and len(by_job)==len(jobs)
    done=set(packet['excluded_conditional_job_ids'])
    for record in records:
        natural=record['natural'];f=record['frozen'];ids=natural['action_ids']
        d.e.accepted._checked_action(ids,natural['stop_reason'],d.CAP)
        assert tokenizer.decode(ids,skip_special_tokens=False)==natural['text']
        assert d.native_record(natural['text'],f['case'],f['golden'],natural['stop_reason'])==natural['parsed']
    for row in rows:
        job=by_job[row['job_id']];f=by_image[row['image_id']]['frozen'];ids=row['free_token_ids']
        assert row['job_id'] not in done;done.add(row['job_id'])
        prefix=job['h_token_ids']+job['c_token_ids']
        assert row['remaining_budget']==d.CAP-len(prefix)
        d.e.accepted._checked_action(ids,row['stop_reason'],row['remaining_budget'])
        assert tokenizer.decode(ids,skip_special_tokens=False)==row['free_text']
        ledger=d.continuation_ledger(job['h_text']+job['c_text'],row['free_text'],f,len(prefix),len(ids),row['stop_reason'])
        assert all(row[k]==v for k,v in ledger.items())
    # Regenerate the deterministic nomination list to cover interrupted per-image
    # nomination writes; never use c outcomes to replace a candidate.
    all_jobs=[]
    for record in records:
        nominated,_=d.nominate(record['frozen'],record['natural'],tokenizer)
        for job in nominated:
            if job['job_id'] in by_job:assert by_job[job['job_id']]==job
        all_jobs.extend(nominated)
    pending=[j for j in all_jobs if j['job_id'] not in done]
    missing=[i for i in pool['image_ids'] if i not in by_image]
    payload={'status':'cold_validated_durable_prefixes_not_successful_attempt','old_packet':d.e.binding(OLD_PACKET),'pool':packet['pool'],'sources':sources,'terminals':terminals,'records':records,'jobs':all_jobs,'completed_conditional_ids':sorted(done),'pending_known_jobs':pending,'missing_natural_ids':missing,'counts':{'records':len(records),'durable_conditional_rows':len(rows),'prior_slice_conditional':2,'missing_natural':len(missing),'pending_known_conditional':len(pending)},'interrupted_attempt_cost':{k:sum(t[k] for t in terminals) for k in ['model_forwards','image_forwards','new_tokens','model_loads']},'wasted_uncommitted_forwards':sum(t['model_forwards']-t['new_tokens'] for t in terminals)}
    d.e.publish(ROOT/'snapshot.json',payload)
    print(payload['counts'])


def previous_records(packet):
    snap=d.e.read(packet['snapshot']['path']);assert d.e.binding(packet['snapshot']['path'])==packet['snapshot']
    records={r['image_id']:r for r in snap['records']}
    for source in packet.get('recovery_slice_images',[]):
        assert d.e.binding(source['path'])==source
        for record in d.e.read_jsonl(source['path']):records[record['image_id']]=record
    return records


def prepare(stage):
    snap=d.e.read(ROOT/'snapshot.json');old=d.e.read(OLD_PACKET);pool=d.e.read(old['pool']['path'])
    packet={**old,'stage':'recovery_'+stage,'snapshot':d.e.binding(ROOT/'snapshot.json'),'recovery_producer':d.e.binding(SCRIPT),'old_attempt':d.e.binding(OLD_PACKET)}
    if stage=='slice':
        first=snap['pending_known_jobs'][0]
        packet.update(image_ids=[snap['missing_natural_ids'][0],first['image_id']],physical_gpus=[0,1],conditional_job_ids=[first['job_id']],excluded_conditional_job_ids=snap['completed_conditional_ids'])
    else:
        receipt=d.e.read(ROOT/'slice/consumer.json')
        packet['recovery_slice_consumer']=d.e.binding(ROOT/'slice/consumer.json')
        packet['recovery_slice_producer_snapshot']=d.e.binding(ROOT/'recovery-slice-producer.py')
        packet['recovery_slice_images']=[d.e.binding(ROOT/f'slice/shard-{s}/images.jsonl') for s in range(2)]
        records=previous_records(packet)
        excluded=set(snap['completed_conditional_ids'])|{x for x in receipt['completed_call_ids'] if not x.startswith('natural:')}
        # Include slice-new image if it has newly nominated conditional jobs.
        pending_images={j['image_id'] for j in snap['pending_known_jobs'] if j['job_id'] not in excluded}
        for source in packet['recovery_slice_images']:
            for r in d.e.read_jsonl(source['path']):
                if r['nominations']:pending_images.add(r['image_id'])
        todo=[i for i in pool['image_ids'] if i not in records or i in pending_images]
        by_pool={i['image_id']:i for i in pool['items']}
        # LPT count proxy balances execution only. Original pool order is untouched.
        bins=[[] for _ in range(8)];loads=[0]*8
        for image_id in sorted(todo,key=lambda i:(-by_pool[i]['object_count'],pool['image_ids'].index(i))):
            rank=min(range(8),key=lambda s:(loads[s],s));bins[rank].append(image_id);loads[rank]+=by_pool[image_id]['object_count']
        # Explicit per-rank ownership avoids modulo aliasing entirely.
        packet.update(image_ids=todo,assigned_image_ids=bins,assignment_object_counts=loads,physical_gpus=list(range(8)),conditional_job_ids=None,excluded_conditional_job_ids=sorted(excluded))
    packet['excluded_completed_call_ids']=old['excluded_completed_call_ids']+[f'natural:{r["image_id"]}' for r in snap['records'] if r['source']=='new_train']+[x for x in snap['completed_conditional_ids'] if x not in old['excluded_conditional_job_ids']]
    if stage=='full':packet['excluded_completed_call_ids']+=receipt['completed_call_ids']
    d.e.publish(ROOT/f'{stage}-packet.json',packet)
    print(d.e.binding(ROOT/f'{stage}-packet.json'))


def worker(packet_path,output,shard):
    packet=d.e.read(packet_path);assert d.e.binding(SCRIPT)==packet['recovery_producer']
    records=previous_records(packet)
    original=d.run_items
    def items(pool,p):
        values=original(pool,p)
        for image_id,record in records.items():
            values[image_id]={**values[image_id],'source':'reused_recovery','frozen':record['frozen'],'natural':record['natural']}
        return values
    d.run_items=items;d.GPUS=tuple(range(8))
    if packet.get('assigned_image_ids'):
        # d.worker strides its packet list. Rank-local packet is a bound
        # scheduling projection, not a changed scientific pool or producer.
        local={**packet,'image_ids':packet['assigned_image_ids'][shard],'physical_gpus':[shard]}
        local_path=ROOT/f'rank-{shard}-packet.json';d.e.publish(local_path,local)
        # Single local rank uses the actual physical index via local GPUS.
        d.GPUS=(shard,);d.worker(local_path,Path(output),0)
    else:d.worker(Path(packet_path),Path(output),shard)


def launch(packet_path,output):
    p=d.e.read(packet_path);output=Path(output);assert not output.exists();output.mkdir()
    processes=[]
    for shard,gpu in enumerate(p['physical_gpus']):
        log=(output/f'shard-{shard}.log').open('x')
        command=[sys.executable,str(SCRIPT),'worker','--packet',str(packet_path),'--output',str(output/f'shard-{shard}'),'--shard',str(shard)]
        proc=subprocess.Popen(command,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',PYTHONPATH=str(d.e.WORKTREE)),stdout=log,stderr=subprocess.STDOUT)
        processes.append((shard,proc,log))
    exits=[]
    for shard,proc,log in processes:exits.append({'shard':shard,'exit_code':proc.wait()});log.close()
    d.e.publish(output/'outer-exits.json',exits);assert all(x['exit_code']==0 for x in exits)


def consume_full():
    """Cold each real rank, then exact-key union with sealed prior producers."""
    packet=d.e.read(ROOT/'full-packet.json');pool=d.e.read(packet['pool']['path'])
    exits=d.e.read(ROOT/'full/outer-exits.json')
    assert len(exits)==8 and all(x['exit_code']==0 for x in exits)
    rank_receipts=[]
    for rank in range(8):
        # Read-only symlink exposes the existing nearest consumer's one-rank
        # interface; this is an explicit cold projection, not a fake GPU run.
        cold=ROOT/f'cold-rank-{rank}';cold.mkdir()
        (cold/'shard-0').symlink_to(ROOT/f'full/shard-{rank}',target_is_directory=True)
        d.e.publish(cold/'outer-exits.json',[{'shard':0,'exit_code':exits[rank]['exit_code']}])
        d.e.publish(cold/'projection.json',{'role':'cold_consumer_projection_only','actual_outer_exits':d.e.binding(ROOT/'full/outer-exits.json'),'physical_gpu':rank,'actual_shard_root':str(ROOT/f'full/shard-{rank}'),'packet':d.e.binding(ROOT/f'rank-{rank}-packet.json')})
        d.consume(ROOT/f'rank-{rank}-packet.json',cold)
        rank_receipts.append(d.e.binding(cold/'consumer.json'))
    snapshot=d.e.read(ROOT/'snapshot.json')
    by_image={r['image_id']:r for r in snapshot['records']}
    by_job={j['job_id']:j for j in snapshot['jobs']}
    by_result={};bindings=[]
    roots=[d.ROOT/f'slice-v2/shard-{s}' for s in range(4)]+[SOURCE/f'shard-{s}' for s in range(4)]+[ROOT/f'slice/shard-{s}' for s in range(2)]+[ROOT/f'full/shard-{s}' for s in range(8)]
    natural_calls=[]
    for root in roots:
        for name in ('images','jobs','rows'):
            path=root/f'{name}.jsonl';values,rejected=complete_lines(path)
            bindings.append({'binding':d.e.binding(path),'rejected_partial_tail':rejected})
            for value in values:
                if name=='images':
                    key=value['image_id']
                    if key in by_image:assert by_image[key]['natural']==value['natural']
                    else:by_image[key]=value
                    if value['source']=='new_train':natural_calls.append(key)
                elif name=='jobs':
                    key=value['job_id']
                    if key in by_job:assert by_job[key]==value
                    else:by_job[key]=value
                else:
                    key=value['job_id'];assert key not in by_result,'completed conditional call repeated'
                    by_result[key]=value
    assert set(by_image)==set(pool['image_ids']) and len(by_image)==4096
    assert len(natural_calls)==len(set(natural_calls))==3766
    assert set(natural_calls)==set(pool['new_ids'])
    assert set(by_job)==set(by_result),'missing or extra conditional outcome'
    # Preserve canonical frozen pool order for downstream distinct-image-first
    # admission; execution scheduling never becomes selection priority.
    position={image_id:i for i,image_id in enumerate(pool['image_ids'])}
    job_ids=sorted(by_job,key=lambda key:(position[by_job[key]['image_id']],by_job[key]['history_index'],by_job[key]['candidate_index']))
    cards=[]
    for source in [d.ROOT/'slice-v2/consumer.json',ROOT/'slice/consumer.json']+[Path(x['path']) for x in rank_receipts]:
        cards.extend(d.e.read(source)['cards'])
    # Sealed original shards2/3 cards are owned by physical admission; do not
    # rewrite them. Remaining original rows retain exact geometry/result source
    # pointers for that same reviewer-owned projection.
    card_ids={c['job_id'] for c in cards}
    summary={'status':'finite_acquisition_complete_physical_admission_pending','schema':'owner_successor_scale.recovery_union.v1','pool':packet['pool'],'recovery_packet':d.e.binding(ROOT/'full-packet.json'),'interrupted_snapshot':d.e.binding(ROOT/'snapshot.json'),'source_bindings':bindings,'rank_consumers':rank_receipts,'images':4096,'new_natural_calls':3766,'reused_anchor_natural':330,'conditional_calls':len(by_result),'unique_call_ids_verified':True,'pending_natural':0,'pending_conditional':0,'canonical_ordered_job_ids':job_ids,'local_w_status_counts':dict(Counter(r['local_w']['status'] for r in by_result.values())),'cards':cards,'prior_outcome_job_ids_needing_existing_physical_projection':[key for key in job_ids if key not in card_ids],'prior_projection_boundary':'Original sealed2/3 physical-admission outputs stay reviewer-owned; original0/1 durable rows were cold validated in snapshot. No outcome is erased by recovery.','claim_boundary':'Machine candidate only. Admission requires independent singleton c and immediate w review; original no-w/group/uncertain outcomes are not replaced.'}
    d.e.publish(ROOT/'ordered-jobs.json',{'pool':packet['pool'],'jobs':[by_job[key] for key in job_ids]})
    d.e.publish(ROOT/'ordered-results.json',{'pool':packet['pool'],'rows':[by_result[key] for key in job_ids]})
    d.e.publish(ROOT/'image-records.json',{'pool':packet['pool'],'records':[by_image[i] for i in pool['image_ids']]})
    d.e.publish(ROOT/'completion.json',summary)
    print({k:summary[k] for k in ('status','images','new_natural_calls','conditional_calls','local_w_status_counts')})


def main():
    p=argparse.ArgumentParser();p.add_argument('command');p.add_argument('--packet');p.add_argument('--output');p.add_argument('--shard',type=int);a=p.parse_args()
    if a.command=='snapshot':snapshot()
    elif a.command in ('prepare-slice','prepare-full'):prepare(a.command.split('-')[1])
    elif a.command=='worker':worker(a.packet,a.output,a.shard)
    elif a.command=='launch':launch(a.packet,a.output)
    elif a.command=='consume-slice':d.consume(Path(a.packet),Path(a.output))
    elif a.command=='consume-full':consume_full()


if __name__=='__main__':main()
