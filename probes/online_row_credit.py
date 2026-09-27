"""Resident fresh online row credit. The research unit owns all runtime releases."""
from __future__ import annotations

import argparse
from pathlib import Path

from probes import rollout_row_credit as r
from probes import iterative_positive as p
from probes.hidden_human_recovery import negative_evidence
from src.eval.saved_rows import iou_xyxy, one_to_one_matches

ROOT = p.ROOT.parent / 'online-row-credit-01'
RETAINED = r.ROOT / 'cpu-04/retained-10.json'
INPUTS = r.ROOT / 'retained-sft-01/inputs.json'
ENCODINGS = r.ROOT / 'retained-sft-01/encodings.json'
RELEASE_SHA = 'b5354c24780367c65249e4a6b6df21e71054d73a593a15c33c573785406b1664'


def identity(value):
    return p.stable(p.canonical(value))


def verify_producer(records, producer, image_ids):
    assert len(records) == len(image_ids) == len(set(image_ids))
    assert {x['image_id'] for x in records} == set(image_ids)
    assert producer['kind'] in ('live_online', 'historical_CPU_fixture')
    assert isinstance(producer['update'], int) and producer['update'] >= 0
    for record in records:
        assert record['producer'] == producer, 'stale or mixed producer'
        assert record['generated_tokens'] == len(record['token_ids']) <= 3084
        assert record['raw_identity'] == identity({k: record[k] for k in
            ('producer','request_id','image_id','token_ids','text','prompt_token_ids','media_sha256','image_grid_thw','stop_reason')}), 'raw identity drift'


def seal(record, producer):
    record = dict(record, producer=producer)
    record['raw_identity'] = identity({k: record[k] for k in
        ('producer','request_id','image_id','token_ids','text','prompt_token_ids','media_sha256','image_grid_thw','stop_reason')})
    return record


def observations(record, tokenizer):
    encoded = r.aligned_tokens(record, tokenizer)
    parsed = r.parse(record)
    negatives = negative_evidence([record], tokenizer)
    rows = [dict(o, valid=True) for o in parsed.predictions]
    for n in negatives['complete_geometry_invalid']:
        drop = n['parser_drop']
        rows.append(dict(drop, valid=False, description=drop['raw_text'].split('<|object_ref_start|>',1)[1].split('<|object_ref_end|>',1)[0], coord_bins=n['coordinate_bins']))
    coordinate_ids = {tokenizer.convert_tokens_to_ids(f'<|coord_{i}|>'): i for i in range(1000)}
    result = []; seen = set()
    for row in sorted(rows, key=lambda x: x['generated_order']):
        positions = r.row_positions(row, encoded)
        coord = [j for j in positions if record['token_ids'][j] in coordinate_ids]
        assert len(coord) == 4 and coord == list(range(coord[0],coord[0]+4))
        assert [coordinate_ids[record['token_ids'][j]] for j in coord] == row['coord_bins']
        key = p.canonical([row['description'], row['coord_bins']])
        result.append(dict(order=row['generated_order'], key=key, positions=positions,
            coordinate_positions=coord, bbox=row['coord_bins'], description=row['description'], valid=row['valid'], first=key not in seen))
        seen.add(key)
    return result, negatives['malformed_or_censored']


def semantic_site(positive, negative, tokenizer):
    """Both alternatives must disagree on semantics at the same literal prefix."""
    schema = {tokenizer.convert_tokens_to_ids(s) for s in
        ('<|object_ref_start|>','<|object_ref_end|>','<|box_start|>','<|box_end|>','<|im_end|>')}
    for d, (good, bad) in enumerate(zip(positive, negative)):
        if good != bad:
            if good in schema or bad in schema:
                raise ValueError('first divergence is schema, not semantic')
            return dict(offset=d, good=good, bad=bad)
    raise ValueError('no shared-prefix semantic divergence')


def credit(image, record, tokenizer, producer):
    verify_producer([record], producer, [image['image_id']])
    assert record['crop'] == [0,0,image['width'],image['height']]
    rows, malformed = observations(record, tokenizer)
    first_valid = [o for o in rows if o['valid'] and o['first']]
    refs = [dict(owner_id=str(o['coco_ann_id']),reference_coord_bins_1000=o['bbox_2d']) for o in image['objects']]
    predictions = [dict(prediction_id=str(o['order']),generated_order=o['order'],coord_bins_1000=o['bbox']) for o in first_valid]
    matches = one_to_one_matches(refs, predictions, .5)
    by = {o['order']: o for o in rows}; matched = {}; disagreements = []
    for match in matches:
        row = by[int(match['prediction_id'])]; obj = image['objects'][match['reference_index']]
        if row['description'] == obj['desc']:
            matched[row['order']] = obj
        else:
            disagreements.append(dict(order=row['order'], annotation_id=obj['coco_ann_id']))
    positives = [dict(o, annotation_id=matched[o['order']]['coco_ann_id']) for o in first_valid if o['order'] in matched]
    redirect = None; events = []
    canonical = sorted(image['objects'], key=lambda o:(o['bbox_2d'][0],o['bbox_2d'][1],o['coco_ann_id']))
    for duplicate in (o for o in rows if not o['first']):
        prior = [o for o in rows if o['order'] < duplicate['order']]
        veto = dict(overlap=0,literal=0,schema=0); alternatives = []
        for later in first_valid:
            if later['order'] > duplicate['order'] and later['order'] in matched:
                alternatives.append((matched[later['order']], later, 'later_matched_first'))
        alternatives += [(o, None, 'retained_repair') for o in canonical]
        selected = None
        for obj, later, source in alternatives:
            if any(o['valid'] and o['description']==obj['desc'] and iou_xyxy(o['bbox'],obj['bbox_2d'])>=.5 for o in prior):
                veto['overlap'] += 1; continue
            if later is None:
                ids = tokenizer.encode(r.render_row(image,obj).assistant_content_text,add_special_tokens=False)
                description, box = obj['desc'],obj['bbox_2d']
            else:
                ids = [record['token_ids'][j] for j in later['positions']]
                description, box = later['description'],later['bbox']
            if p.canonical([description,box]) in {o['key'] for o in prior}:
                veto['literal'] += 1; continue
            bad = [record['token_ids'][j] for j in duplicate['positions']]
            try: site = semantic_site(ids,bad,tokenizer)
            except ValueError:
                veto['schema'] += 1; continue
            cut = duplicate['positions'][0]
            selected = dict(annotation_id=obj['coco_ann_id'],source=source,description=description,bbox=box,
                token_ids=ids,negative_ids=bad,prefix_cut=cut,positions=list(range(cut,cut+len(ids))),
                site=site,duplicate_order=duplicate['order'])
            break
        events.append(dict(order=duplicate['order'],eligible=selected is not None,veto=veto))
        if selected is not None:
            redirect = selected; break
    return dict(image_id=image['image_id'],producer=producer,raw_identity=record['raw_identity'],M=positives,
        redirect=redirect,observations=rows,redirect_events=events,malformed=malformed,category_disagreements=disagreements,
        complete_rows=len(rows),literal_repeats=sum(not o['first'] for o in rows),invalid=sum(not o['valid'] for o in rows))


def redirect_sequence(image, record, target, tokenizer):
    cut = target['prefix_cut']
    history = record['token_ids'][:cut] + target['token_ids']
    assert history[:cut] == record['token_ids'][:cut]
    assert semantic_site(target['token_ids'],target['negative_ids'],tokenizer) == target['site']
    return r.positive_sequence(image,dict(record,token_ids=history),target,tokenizer)


def legal_slots(row):
    return [(row['coordinate_positions'][j], 0 if j<2 else row['bbox'][j-2]+1, 999 if j<2 else 1000)
            for j in range(4) if j<2 or row['bbox'][j-2]<999]


def trace_positions(plan, record):
    n = len(record['prompt_token_ids'])
    targets = {n+j for row in plan['M'] for j in row['positions']}
    targets.update(n+j for row in plan['observations'] for j,_,_ in legal_slots(row))
    return tuple(sorted(t-1 for t in targets)) or (n-1,)


def legal_objective(logits, positions, rows, prompt_length, coordinate_ids):
    import torch
    lookup = {v:i for i,v in enumerate(positions)}
    assert len(lookup)==len(positions) and len(coordinate_ids)==1000
    losses = []
    for row in rows:
        slots = []
        for pos,lo,hi in legal_slots(row):
            z = logits[0,lookup[prompt_length+pos-1]].float()
            slots.append(torch.logsumexp(z,0)-torch.logsumexp(z[list(coordinate_ids[lo:hi])],0))
        assert slots
        losses.append(torch.stack(slots).mean())
    return torch.stack(losses).mean() if losses else logits.sum()*0


def trace_objective(logits, positions, plan, record, image, tokenizer, vocab):
    import torch
    assert tuple(positions)==trace_positions(plan,record), 'wrong causal positions'
    assert plan['raw_identity']==record['raw_identity'] and plan['producer']==record['producer']
    values = [p.image_loss(logits,r.positive_sequence(image,record,row,tokenizer),vocab,positions)[0] for row in plan['M']]
    m = torch.stack(values).mean() if values else logits.sum()*0
    legal = legal_objective(logits,positions,plan['observations'],len(record['prompt_token_ids']),vocab.coordinate)
    return m+legal, dict(M=m,legal=legal)


def redirect_objective(logits, positions, sequence, target, prompt_length, vocab):
    import torch.nn.functional as F
    assert tuple(positions)==tuple(a.causal_logits_position for a in sequence.atoms)
    d = target['site']['offset']; pos = prompt_length+target['prefix_cut']+d-1
    assert target['token_ids'][:d]==target['negative_ids'][:d]
    assert target['token_ids'][d]==target['site']['good'] and target['negative_ids'][d]==target['site']['bad']
    atom = sequence.atoms[d]
    assert atom.token_type in ('desc_text','coordinate') and atom.causal_logits_position==pos
    z = logits[0,positions.index(pos)].float()
    margin = F.softplus(1+z[target['site']['bad']]-z[target['site']['good']])
    positive,_ = p.image_loss(logits,sequence,vocab,positions)
    return positive+margin, dict(redirect_positive=positive,redirect_margin=margin)


def jobs(image_ids, rank, plans):
    assert len(image_ids)==len(set(image_ids))==18 and 0<=rank<8
    result = []
    for i in sorted(image_ids)[rank::8]:
        result += [dict(image_id=i,branch=b,weight=8/18) for b in
                   (['trace','redirect','R'] if plans[i]['redirect'] else ['trace','R'])]
    return [dict(x,sync=j==len(result)-1) for j,x in enumerate(result)]


def retained_credit(logits, sequence, row_ids, vocab, positions):
    value, rows = r.retained_objective(logits,sequence,row_ids,vocab,positions)
    return .25*value, {'R_unweighted':value,'R_weighted':.25*value}, rows


def diagnostics(branches, logits):
    import torch
    result = {}
    for name,value in branches.items():
        grad, = torch.autograd.grad(value,logits,retain_graph=True)
        assert torch.isfinite(grad).all()
        grad = grad.detach().float()
        result[name] = dict(loss=float(value.detach()),l2=float(grad.norm()),linf=float(grad.abs().max()),
                           support_rows=int((grad.abs().sum(-1)>0).sum()))
    return result


def native_batch(q, item):
    batch = p.native_request(item,p.load(p.POLICY),q.processor)
    assert list(batch.prompt_token_ids[0])==item['prompt_token_ids']
    assert list(batch.image_grids[0])==item['image_grid_thw'] and batch.media_sha256[0]==item['media_sha256']
    return batch


def forward(q, model, batch, image, record, plan, encoding, vocab, branch):
    import torch
    from src.qwen.native import exact_history_inputs
    if branch=='R':
        _,sequence,row_ids = r.retained_sequence(image,q)
        assert list(sequence.input_ids)==encoding['input_ids']
        assert [a.to_artifact_dict() for a in sequence.atoms]==encoding['atoms']
        full = sequence.input_ids; positions = tuple(a.causal_logits_position for a in sequence.atoms)
    elif branch=='redirect':
        sequence = redirect_sequence(image,record,plan['redirect'],q.tokenizer)
        full = sequence.input_ids; positions = tuple(a.causal_logits_position for a in sequence.atoms)
    else:
        assert branch=='trace'
        full = record['prompt_token_ids']+record['token_ids']; positions = trace_positions(plan,record)
    assert len(full)<=p.MAX_LENGTH
    kwargs = exact_history_inputs(q.model,batch.inputs,[full],pad_token_id=q.tokenizer.pad_token_id)
    kwargs['logits_to_keep'] = torch.tensor(positions,device='cuda')
    with torch.autocast('cuda',dtype=torch.bfloat16): logits = model(**kwargs).logits
    if branch=='R':
        loss,terms,rows = retained_credit(logits,sequence,row_ids,vocab,positions)
    elif branch=='redirect':
        loss,terms = redirect_objective(logits,positions,sequence,plan['redirect'],len(record['prompt_token_ids']),vocab)
        rows = []
    else:
        loss,terms = trace_objective(logits,positions,plan,record,image,q.tokenizer,vocab); rows = []
    comparison = []
    if branch=='trace' and record.get('raw_logprobs'):
        n = len(record['prompt_token_ids'])
        for j,pos in enumerate(positions):
            index = pos+1-n
            if 0<=index<len(record['token_ids']):
                z = logits[0,j].detach().float()
                lp = float(z.log_softmax(-1)[record['token_ids'][index]])
                comparison.append(dict(index=index,cached_logp=record['raw_logprobs'][index],replay_logp=lp,
                                       difference=lp-record['raw_logprobs'][index]))
                if len(comparison)==4: break
    return loss,dict(image_id=image['image_id'],branch=branch,producer=plan['producer'],raw_identity=record['raw_identity'],
        tokens=len(full),visual_tokens=int(__import__('math').prod(record['image_grid_thw'])//4),
        input_sha256=identity(list(full)),positions=list(positions),loss=float(loss.detach()),
        terms={k:float(v.detach()) for k,v in terms.items()},logit_derivatives=diagnostics(terms,logits),
        row_losses=rows,cached_replay=comparison,logits_sha256=p.tensor_hash(logits))


def parameter_identity(model):
    return {name:p.tensor_hash(value) for name,value in model.named_parameters() if value.requires_grad}


def start(output, root):
    import os,time,torch
    from src.artifacts.git_identity import capture_source_identity
    for path,sha in p.load(root/'qualification.json')['sha256'].items(): assert p.digest(path)==sha,path
    assert int(os.environ['WORLD_SIZE'])==8
    rank = int(os.environ['RANK']);torch.cuda.set_device(int(os.environ['LOCAL_RANK']));torch.manual_seed(92711)
    sources = ['probes/online_row_credit.py','probes/rollout_row_credit.py','probes/iterative_positive.py','probes/hidden_human_recovery.py',
               *sorted(str(x) for x in Path('src').rglob('*.py'))]
    source = capture_source_identity(sources)
    out = output/f'rank-{rank}';out.mkdir(parents=True,exist_ok=False)
    p.write(out/'entry.json',dict(pid=os.getpid(),rank=rank,start=time.time(),source=source))
    return rank,out,sources,source


def run(output, root, updates=1):
    import os,time,math,torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from src.qwen.generation import generate_continuations,NativeGenerationPolicy
    from src.losses.vocab import build_token_vocabulary_groups
    from src.artifacts.git_identity import verify_source_identity
    assert updates in (1,64)
    rank,out,sources,source = start(output,root); begin=time.monotonic()
    images={x['image_id']:x for x in p.load(RETAINED)}
    inputs={x['image_id']:x for x in p.load(INPUTS)}
    encodings={x['image_id']:x for x in p.load(ENCODINGS)}
    assert set(images)==set(inputs)==set(encodings) and len(images)==18
    local=sorted(images)[rank::8];previous_metrics=None
    dist.init_process_group('nccl')
    q,delta,composition=p.compose(Path(p.load(p.POLICY)['checkpoint']),evaluation=False)
    p.write(out/'composition.json',composition)
    assert q.model.config.text_config.attention_dropout==0
    dropouts={name:getattr(module,'p',0) for name,module in q.model.named_modules() if 'lora_dropout' in name}
    assert all(value==0 for value in dropouts.values())
    p.write(out/'online-policy.json',dict(attention_dropout=q.model.config.text_config.attention_dropout,lora_dropout=dropouts,
        base='BF16',trainable='FP32',attention='flash_attention_2',autocast='BF16 generation and replay',temperature=0,top_p=1,top_k=0,
        repetition_penalty=1,max_new_tokens=3084,use_model_defaults=False,seed=92711))
    vocab=build_token_vocabulary_groups(q.token_identity,tokenizer=q.tokenizer)
    assert tuple(vocab.coordinate)==tuple(q.tokenizer.convert_tokens_to_ids(f'<|coord_{j}|>') for j in range(1000))
    params=[x for x in q.model.parameters() if x.requires_grad]; flags={n:x.requires_grad for n,x in q.model.named_parameters()}
    delta_ids={id(x) for x in delta.delta_tensors().values()}
    optimizer=torch.optim.AdamW([dict(params=[x for x in params if id(x) not in delta_ids],lr=1e-5),
        dict(params=list(delta.delta_tensors().values()),lr=5e-6)],betas=(.9,.999),eps=1e-8,weight_decay=0)
    q.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});q.model.enable_input_require_grads()
    model=DistributedDataParallel(q.model,device_ids=[int(os.environ['LOCAL_RANK'])],broadcast_buffers=False)
    batches={i:native_batch(q,inputs[i]) for i in local}
    if rank==0:p.save_checkpoint(q,delta,output/'checkpoint-0')
    dist.barrier()
    for version in range(updates+1):
        fingerprint=parameter_identity(q.model); hashes=[None]*8
        dist.all_gather_object(hashes,identity(fingerprint));assert len(set(hashes))==1
        producer=dict(kind='live_online',update=version,parameter_sha256=hashes[0],source=source['commit'] if 'commit' in source else identity(source))
        p.write(out/f'producer-{version}.json',dict(producer=producer,parameters=fingerprint))
        q.model.eval();records=[];before=time.monotonic()
        for i in local:
            generation_start=time.monotonic()
            item=dict(inputs[i],arm='greedy',seed=92711,temperature=0)
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                result=generate_continuations(q.model,batches[i],extensions=[()],budgets=[3084],
                    eos_token_id=q.tokenizer.convert_tokens_to_ids('<|im_end|>'),pad_token_id=q.tokenizer.pad_token_id,
                    policy=NativeGenerationPolicy(temperature=0,top_p=1,top_k=0,repetition_penalty=1,use_model_defaults=False),
                    trace='raw_and_policy' if version==0 and i==min(images) else 'none',seed=None)[0]
            record=seal(dict(item,token_ids=list(result.token_ids),text=q.tokenizer.decode(result.token_ids,skip_special_tokens=False),
                generated_tokens=len(result.token_ids),stop_reason=result.stop_reason,generation_seconds=time.monotonic()-generation_start,raw_logprobs=result.raw_logprobs),producer)
            records.append(record)
        # No rank updates until every same-version trajectory is saved and validated.
        directory=output/f'rollout-{version}'/f'rank-{rank}';directory.mkdir(parents=True,exist_ok=False)
        for record in records:p.write(directory/f"{record['image_id']}.json",record)
        p.write(directory/'complete.json',dict(status='complete',source=source,producer=producer,
            artifacts={x.name:p.digest(x) for x in directory.glob('*.json')}))
        gathered=[None]*8;dist.all_gather_object(gathered,records)
        all_records=[x for part in gathered for x in part];verify_producer(all_records,producer,sorted(images))
        assert parameter_identity(q.model)==fingerprint
        assert flags=={n:x.requires_grad for n,x in q.model.named_parameters()}
        plans={x['image_id']:credit(images[x['image_id']],x,q.tokenizer,producer) for x in records}
        p.write(out/f'credit-{version}.json',list(plans.values()))
        metrics=r.assess_outputs([images[i] for i in local],[],records)
        if previous_metrics is not None:
            prior={x['image_id']:x for x in previous_metrics}
            for row in metrics:
                row['retained_change']={}
                for mode in ('raw','category'):
                    old=set(prior[row['image_id']]['ids'][mode]['retained']);new=set(row['ids'][mode]['retained'])
                    row['retained_change'][mode]=dict(gained=sorted(new-old),lost=sorted(old-new),preserved=sorted(new&old))
        p.write(out/f'retained-metrics-{version}.json',metrics);previous_metrics=metrics
        if version==updates:break
        by={x['image_id']:x for x in records};q.model.train();optimizer.zero_grad(set_to_none=True)
        evidence=r.accumulate_family_step(model,jobs(list(images),rank,plans),
            lambda job:forward(q,model,batches[job['image_id']],images[job['image_id']],by[job['image_id']],
                plans[job['image_id']],encodings[job['image_id']],vocab,job['branch']))
        norms={n:float(x.grad.float().norm()) if x.grad is not None else None for n,x in q.model.named_parameters() if x.requires_grad}
        assert all(v is not None and math.isfinite(v) for v in norms.values())
        assert all(any(v>0 for n,v in norms.items() if tag in n) for tag in ('lora_','embed_tokens.shared_embed_delta','lm_head.shared_embed_delta'))
        synced=[None]*8;dist.all_gather_object(synced,identity(norms));assert len(set(synced))==1
        total=float(torch.nn.utils.clip_grad_norm_(params,1,error_if_nonfinite=True));optimizer.step()
        assert all(torch.isfinite(x).all() for x in params)
        states=sorted({int(state['step']) for state in optimizer.state.values()});assert states==[version+1]
        p.write(out/f'update-{version+1}.json',dict(update=version+1,producer=producer,forwards=evidence,gradient_norms=norms,
            synchronized_norms=synced,total_norm=total,optimizer_steps=states,optimizer_state_count=len(optimizer.state),lrs=[g['lr'] for g in optimizer.param_groups],seconds=time.monotonic()-before))
        if version+1 in (1,2,4,8,16,32,64):
            if rank==0:p.save_checkpoint(q,delta,output/f'checkpoint-{version+1}')
            dist.barrier()
    verify_source_identity(source,required_paths=sources)
    p.write(out/'complete.json',dict(status='complete',source=source,updates=updates,wall_seconds=time.monotonic()-begin,
        peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),
        artifacts={x.name:p.digest(x) for x in out.glob('*.json')}))
    dist.destroy_process_group()


def frozen_records(root, image_ids, freeze=False):
    """Same shard/hash readback, without predecessor reader's old-visible-label read."""
    records=[]
    for rank in range(8):
        directory=root/f'rank-{rank}';receipt=p.load(directory/'complete.json')
        assert receipt['status']=='complete'
        for name,sha in receipt['artifacts'].items():
            assert p.digest(directory/name)==sha
            records.append(p.load(directory/name))
    assert len(records)==18 and {x['image_id'] for x in records}==set(image_ids)
    if freeze:
        frozen={str(x.relative_to(root)):p.digest(x) for x in sorted(root.rglob('*.json')) if x.name!='frozen.json'}
        if (root/'frozen.json').exists():assert p.load(root/'frozen.json')==frozen
        else:p.write(root/'frozen.json',frozen)
    return sorted(records,key=lambda x:x['image_id'])


def readback(output, root, updates):
    """Fresh process, frozen raw shard readback; no model or evaluator truth."""
    for path,sha in p.load(root/'qualification.json')['sha256'].items():assert p.digest(path)==sha,path
    inputs={x['image_id']:x for x in p.load(INPUTS)};result=[]
    for rank in range(8):
        directory=output/f'rank-{rank}';receipt=p.load(directory/'complete.json');assert receipt['status']=='complete' and receipt['updates']==updates
        for name,sha in receipt['artifacts'].items():assert p.digest(directory/name)==sha
        for row in receipt['source']['files']:assert p.digest(row['path'])==row['sha256']
        for step in range(1,updates+1):
            evidence=p.load(directory/f'update-{step}.json')
            assert evidence['optimizer_steps']==[step] and evidence['optimizer_state_count']==590
            assert evidence['lrs']==[1e-5,5e-6]
            assert len(set(evidence['synchronized_norms']))==1
            assert sum(x['sync'] for x in evidence['forwards'])==1 and evidence['forwards'][-1]['sync']
            assert all(x['image_weight']==8/18 for x in evidence['forwards'])
    for checkpoint in sorted(output.glob('checkpoint-*')):
        for name,sha in p.load(checkpoint/'identity.json').items():assert p.digest(checkpoint/name)==sha
    for version in range(updates+1):
        records=frozen_records(output/f'rollout-{version}',set(inputs),freeze=True)
        producer=records[0]['producer'];verify_producer(records,producer,list(inputs));assert producer['update']==version and producer['kind']=='live_online'
        for rank in range(8):
            bound=p.load(output/f'rank-{rank}'/f'producer-{version}.json')
            assert bound['producer']==producer and identity(bound['parameters'])==producer['parameter_sha256']
        if result:assert producer['parameter_sha256']!=result[-1]['producer']['parameter_sha256']
        for record in records:
            for key in inputs[record['image_id']]:assert record[key]==inputs[record['image_id']][key],key
        result.append(dict(update=version,producer=producer,requests=len(records),tokens=sum(x['generated_tokens'] for x in records),
            frozen_sha256=p.digest(output/f'rollout-{version}/frozen.json')))
    p.write(output/'readback.json',result)


def offline(output, root, updates):
    """Only this separate process opens evaluator truth, after all raw outputs freeze."""
    for path,sha in p.load(root/'qualification.json')['sha256'].items():assert p.digest(path)==sha,path
    read=p.load(output/'readback.json')
    assert [x['update'] for x in read]==list(range(updates+1))
    images=p.load(RETAINED);frozen={}
    for row in read:
        directory=output/f"rollout-{row['update']}"
        records=frozen_records(directory,[x['image_id'] for x in images],freeze=True)
        assert p.digest(directory/'frozen.json')==row['frozen_sha256']
        frozen['zero' if row['update']==0 else str(row['update'])]=records
    p.write(output/'offline-inputs-frozen.json',read)
    for path,sha in p.load(root/'qualification.json')['evaluator_sha256'].items():assert p.digest(path)==sha,path
    partitions=p.load(r.ROOT/'cpu-03/evaluator-partitions.json')
    assert p.digest(r.TRUTH)==partitions['truth_sha256']
    truth=p.load(r.TRUTH)
    scored={k:r.assess_outputs(truth,partitions['hidden10'],v) for k,v in frozen.items()}
    p.write(output/'offline-results.json',r.family_outcomes(scored,[]))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('run','readback','offline'))
    parser.add_argument('--root',type=Path,default=ROOT);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--updates',type=int,choices=(1,64),default=1)
    a=parser.parse_args();{'run':run,'readback':readback,'offline':offline}[a.command](a.output,a.root,a.updates)


if __name__=='__main__':main()
