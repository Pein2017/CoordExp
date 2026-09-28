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


def erroneous_slots(rows):
    errors={}
    for row in rows:
        for pos,lo,hi in legal_slots(row):
            value=row['bbox'][row['coordinate_positions'].index(pos)]
            if not lo<=value<hi:
                item=(pos,lo,hi)
                assert pos not in errors or errors[pos]==item
                errors[pos]=item
    return tuple(errors[k] for k in sorted(errors))


def max_geometry_margin(z, legal_ids):
    import torch
    import torch.nn.functional as F
    z=z.float();mask=torch.ones(z.shape[-1],dtype=torch.bool,device=z.device)
    mask[list(legal_ids)]=False
    return F.softplus(1+torch.amax(z[mask])-torch.amax(z[list(legal_ids)]))


def greedy_geometry_objective(logits, positions, rows, prompt_length, coordinate_ids):
    import torch
    assert len(set(positions))==len(positions)
    values=[max_geometry_margin(logits[0,positions.index(prompt_length+pos-1)],coordinate_ids[lo:hi])
            for pos,lo,hi in erroneous_slots(rows)]
    return torch.stack(values).mean() if values else logits.sum()*0


def trace_objective(logits, positions, plan, record, image, tokenizer, vocab, geometry_weight=0):
    import torch
    assert tuple(positions)==trace_positions(plan,record), 'wrong causal positions'
    assert plan['raw_identity']==record['raw_identity'] and plan['producer']==record['producer']
    values = [p.image_loss(logits,r.positive_sequence(image,record,row,tokenizer),vocab,positions)[0] for row in plan['M']]
    m = torch.stack(values).mean() if values else logits.sum()*0
    legal = legal_objective(logits,positions,plan['observations'],len(record['prompt_token_ids']),vocab.coordinate)
    assert geometry_weight in (0,.1)
    if geometry_weight:
        g=greedy_geometry_objective(logits,positions,plan['observations'],len(record['prompt_token_ids']),vocab.coordinate)
        return m+legal+geometry_weight*g,dict(M=m,legal=legal,Gmax_unweighted=g,Gmax_weighted=geometry_weight*g)
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


def preservation_entry(image, record, tokenizer):
    """Frozen incoming predictions, selected without evaluator truth."""
    from src.eval.detection_categories import COCO_80_CLASS_NAMES
    verify_producer([record],record['producer'],[image['image_id']])
    assert record['producer']['kind']=='live_online' and record['producer']['update']==0
    rows,_=observations(record,tokenizer);selected=[];dispositions=[]
    for row in rows:
        if not row['valid']:reason='invalid'
        elif not row['first']:reason='literal_repeat'
        elif row['description'] not in COCO_80_CLASS_NAMES:reason='out_of_scope'
        else:
            overlaps=[o for o in image['objects'] if iou_xyxy(row['bbox'],o['bbox_2d'])>=.5]
            reason=('cross_category_conflict' if any(o['desc']!=row['description'] for o in overlaps)
                    else 'same_category_withheld' if overlaps else 'eligible')
        dispositions.append(dict(order=row['order'],key=row['key'],reason=reason))
        if reason=='eligible':selected.append(row)
    return dict(image_id=image['image_id'],record=record,rows=selected,dispositions=dispositions)


def preservation_sequences(image, entry, tokenizer):
    assert entry==preservation_entry(image,entry['record'],tokenizer), 'preservation eligibility or identity drift'
    record=entry['record']
    sequences=[r.positive_sequence(image,record,row,tokenizer) for row in entry['rows']]
    positions=tuple(sorted(a.causal_logits_position for seq in sequences for a in seq.atoms))
    assert len(positions)==len(set(positions))
    return sequences,positions or (len(record['prompt_token_ids'])-1,)


def preservation_objective(logits, positions, sequences, vocab, weight):
    import torch
    assert weight in (0,.25)
    expected=tuple(sorted(a.causal_logits_position for seq in sequences for a in seq.atoms))
    assert not expected or tuple(positions)==expected, 'preservation causal positions'
    values=[p.image_loss(logits,seq,vocab,positions)[0] for seq in sequences]
    value=torch.stack(values).mean() if values else logits.sum()*0
    return weight*value,dict(P0_unweighted=value,P0_weighted=weight*value)


def preservation_binding(root, weight, bank_sha):
    assert weight in (0,.25)
    spec=p.load(root/'qualification.json').get('preservation')
    if spec is None:
        assert weight==0 and bank_sha is None
        return None,None
    assert bank_sha==spec['sha256']==p.digest(spec['path']), 'wrong preservation bank identity'
    bank=p.load(spec['path'])
    assert bank['kind']=='fixed_incoming_prediction_preservation'
    return dict(weight=weight,bank_sha256=bank_sha,bank_path=spec['path']),{x['image_id']:x for x in bank['images']}


def jobs(image_ids, rank, plans, preservation_weight=0):
    assert len(image_ids)==len(set(image_ids))==18 and 0<=rank<8
    assert preservation_weight in (0,.25)
    result = []
    for i in sorted(image_ids)[rank::8]:
        result += [dict(image_id=i,branch=b,weight=8/18) for b in
                   (['trace']+(['redirect'] if plans[i]['redirect'] else [])+(['P0'] if preservation_weight else [])+['R'])]
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


def forward(q, model, batch, image, record, plan, encoding, vocab, branch, preservation=None, preservation_weight=0, geometry_weight=0):
    import torch
    from src.qwen.native import exact_history_inputs
    lineage=None
    if branch=='P0':
        sequences,positions=preservation_sequences(image,preservation,q.tokenizer)
        record=preservation['record'];full=record['prompt_token_ids']+record['token_ids']
        lineage=dict(kind='fixed_incoming_preservation',entry_sha256=identity(preservation),weight=preservation_weight,
                     source_producer=record['producer'],source_raw_identity=record['raw_identity'])
    elif branch=='R':
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
    if branch=='P0':
        loss,terms=preservation_objective(logits,positions,sequences,vocab,preservation_weight)
        rows=[dict(order=row['order'],atoms=len(seq.atoms)) for row,seq in zip(preservation['rows'],sequences)]
    elif branch=='R':
        loss,terms,rows = retained_credit(logits,sequence,row_ids,vocab,positions)
    elif branch=='redirect':
        loss,terms = redirect_objective(logits,positions,sequence,plan['redirect'],len(record['prompt_token_ids']),vocab)
        rows = []
    else:
        loss,terms = trace_objective(logits,positions,plan,record,image,q.tokenizer,vocab,geometry_weight); rows = []
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
    evidence=dict(image_id=image['image_id'],branch=branch,producer=plan['producer'],raw_identity=record['raw_identity'],
        tokens=len(full),visual_tokens=int(__import__('math').prod(record['image_grid_thw'])//4),
        input_sha256=identity(list(full)),positions=list(positions),loss=float(loss.detach()),
        terms={k:float(v.detach()) for k,v in terms.items()},logit_derivatives=diagnostics(terms,logits),
        row_losses=rows,cached_replay=comparison,logits_sha256=p.tensor_hash(logits))
    if lineage is not None:evidence['preservation']=lineage
    if branch=='trace' and geometry_weight:
        evidence['geometry']=dict(weight=geometry_weight,error_slots=list(erroneous_slots(plan['observations'])))
    return loss,evidence


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


def export_steps(updates):
    assert updates in (1,16,64)
    return tuple(x for x in (0,1,2,4,8,16,32,64) if x<=updates)


def geometry_binding(root, checkpoint, weight, recipe_sha256):
    assert weight in (0,.1)
    spec=p.load(root/'qualification.json').get('geometry')
    if spec is None:
        assert checkpoint is None and weight==0 and recipe_sha256 is None
        return None
    assert checkpoint is not None and str(checkpoint)==spec['checkpoint'], 'wrong start checkpoint'
    assert recipe_sha256==identity(spec), 'wrong geometry recipe'
    assert p.digest(checkpoint/'identity.json')==spec['checkpoint_identity_sha256']
    for name,sha in p.load(checkpoint/'identity.json').items():assert p.digest(checkpoint/name)==sha,name
    return dict(recipe_sha256=recipe_sha256,checkpoint=str(checkpoint),checkpoint_identity_sha256=spec['checkpoint_identity_sha256'],weight=weight)


def verify_start_export(checkpoint, exported):
    import torch
    from safetensors.torch import load_file
    from src.adapters.dora import normalize_dora_state_key
    for filename in ('adapter/adapter_model.safetensors','special_token_embeddings/special_token_embeddings.safetensors'):
        def read(path):
            return {normalize_dora_state_key(k,adapter_name='default').replace('.lora_magnitude_vector.weight','.lora_magnitude_vector'):v
                    for k,v in load_file(str(path/filename)).items()}
        a,b=read(checkpoint),read(exported)
        assert a.keys()==b.keys()
        assert all(a[k].dtype==b[k].dtype and torch.equal(a[k],b[k]) for k in a), 'zero differs from explicit start'


def run(output, root, updates=1, preservation_weight=0, preservation_bank_sha256=None,
        geometry_weight=0, start_checkpoint=None, recipe_sha256=None):
    import os,time,math,torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from src.qwen.generation import generate_continuations,NativeGenerationPolicy
    from src.losses.vocab import build_token_vocabulary_groups
    from src.artifacts.git_identity import verify_source_identity
    scheduled=export_steps(updates)
    geometry=geometry_binding(root,start_checkpoint,geometry_weight,recipe_sha256)
    if geometry is not None:assert preservation_weight==0 and preservation_bank_sha256 is None
    binding,bank=preservation_binding(root,preservation_weight,preservation_bank_sha256)
    rank,out,sources,source = start(output,root); begin=time.monotonic()
    if binding is not None:p.write(out/'preservation.json',binding)
    if geometry is not None:p.write(out/'geometry.json',geometry)
    images={x['image_id']:x for x in p.load(RETAINED)}
    inputs={x['image_id']:x for x in p.load(INPUTS)}
    encodings={x['image_id']:x for x in p.load(ENCODINGS)}
    assert set(images)==set(inputs)==set(encodings) and len(images)==18
    if bank is not None:assert set(bank)==set(images)
    local=sorted(images)[rank::8];previous_metrics=None
    dist.init_process_group('nccl')
    q,delta,composition=p.compose(start_checkpoint if start_checkpoint is not None else Path(p.load(p.POLICY)['checkpoint']),evaluation=False)
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
        evidence=r.accumulate_family_step(model,jobs(list(images),rank,plans,preservation_weight),
            lambda job:forward(q,model,batches[job['image_id']],images[job['image_id']],by[job['image_id']],
                plans[job['image_id']],encodings[job['image_id']],vocab,job['branch'],
                bank[job['image_id']] if bank is not None else None,preservation_weight,geometry_weight))
        norms={n:float(x.grad.float().norm()) if x.grad is not None else None for n,x in q.model.named_parameters() if x.requires_grad}
        assert all(v is not None and math.isfinite(v) for v in norms.values())
        assert all(any(v>0 for n,v in norms.items() if tag in n) for tag in ('lora_','embed_tokens.shared_embed_delta','lm_head.shared_embed_delta'))
        synced=[None]*8;dist.all_gather_object(synced,identity(norms));assert len(set(synced))==1
        total=float(torch.nn.utils.clip_grad_norm_(params,1,error_if_nonfinite=True));optimizer.step()
        assert all(torch.isfinite(x).all() for x in params)
        states=sorted({int(state['step']) for state in optimizer.state.values()});assert states==[version+1]
        p.write(out/f'update-{version+1}.json',dict(update=version+1,producer=producer,forwards=evidence,gradient_norms=norms,
            synchronized_norms=synced,total_norm=total,optimizer_steps=states,optimizer_state_count=len(optimizer.state),lrs=[g['lr'] for g in optimizer.param_groups],seconds=time.monotonic()-before))
        if version+1 in scheduled:
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


def readback(output, root, updates, preservation_weight=0, preservation_bank_sha256=None,
             geometry_weight=0, start_checkpoint=None, recipe_sha256=None):
    """Fresh process, frozen raw shard readback; no model or evaluator truth."""
    for path,sha in p.load(root/'qualification.json')['sha256'].items():assert p.digest(path)==sha,path
    binding,bank=preservation_binding(root,preservation_weight,preservation_bank_sha256)
    geometry=geometry_binding(root,start_checkpoint,geometry_weight,recipe_sha256)
    if geometry is not None:assert preservation_weight==0 and preservation_bank_sha256 is None
    inputs={x['image_id']:x for x in p.load(INPUTS)};result=[]
    for rank in range(8):
        directory=output/f'rank-{rank}';receipt=p.load(directory/'complete.json');assert receipt['status']=='complete' and receipt['updates']==updates
        if binding is not None:assert p.load(directory/'preservation.json')==binding, 'wrong preservation arm'
        if geometry is not None:assert p.load(directory/'geometry.json')==geometry,'wrong geometry arm'
        for name,sha in receipt['artifacts'].items():assert p.digest(directory/name)==sha
        for row in receipt['source']['files']:assert p.digest(row['path'])==row['sha256']
        for step in range(1,updates+1):
            evidence=p.load(directory/f'update-{step}.json')
            assert evidence['optimizer_steps']==[step] and evidence['optimizer_state_count']==590
            assert evidence['lrs']==[1e-5,5e-6]
            assert len(set(evidence['synchronized_norms']))==1
            assert sum(x['sync'] for x in evidence['forwards'])==1 and evidence['forwards'][-1]['sync']
            assert all(x['image_weight']==8/18 for x in evidence['forwards'])
            if geometry is not None:
                plans={x['image_id']:x for x in p.load(directory/f'credit-{step-1}.json')}
                for row in evidence['forwards']:
                    assert row['branch']!='P0'
                    if row['branch']=='trace':
                        assert ('geometry' in row)==bool(geometry_weight)
                        if geometry_weight:
                            assert row['geometry']==dict(weight=geometry_weight,error_slots=[list(x) for x in erroneous_slots(plans[row['image_id']]['observations'])])
                            assert row['terms']['Gmax_weighted']==float(__import__('torch').tensor(row['terms']['Gmax_unweighted'],dtype=__import__('torch').float32)*geometry_weight)
            if binding is not None:
                selected=[x for x in evidence['forwards'] if x['branch']=='P0']
                local=sorted(inputs)[rank::8]
                assert [x['image_id'] for x in selected]==(local if preservation_weight else [])
                for row in selected:
                    entry=bank[row['image_id']];record=entry['record']
                    assert row['preservation']==dict(kind='fixed_incoming_preservation',entry_sha256=identity(entry),weight=preservation_weight,
                        source_producer=record['producer'],source_raw_identity=record['raw_identity'])
                    assert row['input_sha256']==identity(record['prompt_token_ids']+record['token_ids'])
                    expected=sorted(len(record['prompt_token_ids'])+j-1 for target in entry['rows'] for j in target['positions'])
                    assert row['positions']==(expected or [len(record['prompt_token_ids'])-1])
                    assert row['row_losses']==[dict(order=x['order'],atoms=len(x['positions'])) for x in entry['rows']]
                    assert row['terms']['P0_weighted']==preservation_weight*row['terms']['P0_unweighted']
    scheduled=export_steps(updates)
    checkpoints={output/f'checkpoint-{step}' for step in scheduled}
    assert set(output.glob('checkpoint-*'))==checkpoints, 'missing or unexpected scheduled export'
    for checkpoint in sorted(checkpoints):
        assert checkpoint.is_dir() and (checkpoint/'identity.json').is_file(), checkpoint
        for name,sha in p.load(checkpoint/'identity.json').items():assert p.digest(checkpoint/name)==sha
    if geometry is not None:verify_start_export(start_checkpoint,output/'checkpoint-0')
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


def offline(output, root, updates, preservation_weight=0, preservation_bank_sha256=None,
            geometry_weight=0, start_checkpoint=None, recipe_sha256=None):
    """Only this separate process opens evaluator truth, after all raw outputs freeze."""
    for path,sha in p.load(root/'qualification.json')['sha256'].items():assert p.digest(path)==sha,path
    binding,_=preservation_binding(root,preservation_weight,preservation_bank_sha256)
    geometry=geometry_binding(root,start_checkpoint,geometry_weight,recipe_sha256)
    if geometry is not None:
        assert preservation_weight==0 and preservation_bank_sha256 is None
        for rank in range(8):assert p.load(output/f'rank-{rank}/geometry.json')==geometry
    if binding is not None:
        for rank in range(8):assert p.load(output/f'rank-{rank}/preservation.json')==binding
    read=p.load(output/'readback.json')
    assert [x['update'] for x in read]==list(range(updates+1))
    images=p.load(RETAINED);frozen={}
    for row in read:
        directory=output/f"rollout-{row['update']}"
        records=frozen_records(directory,[x['image_id'] for x in images],freeze=True)
        assert p.digest(directory/'frozen.json')==row['frozen_sha256']
        frozen['zero' if row['update']==0 else str(row['update'])]=records
    p.write(output/'offline-inputs-frozen.json',read)
    if geometry is not None:
        shapes={}
        for k in frozen:
            version=0 if k=='zero' else int(k);values=[]
            for rank in range(8):
                for plan in p.load(output/f'rank-{rank}/credit-{version}.json'):
                    widths=[row['bbox'][2]-row['bbox'][0] for row in plan['observations']]
                    heights=[row['bbox'][3]-row['bbox'][1] for row in plan['observations']]
                    values.append(dict(image_id=plan['image_id'],widths=widths,heights=heights,
                        widths_le1=sum(x<=1 for x in widths),heights_le1=sum(x<=1 for x in heights)))
            shapes[k]=values
        p.write(output/'predicted-shapes.json',dict(scope='All certified complete occurrences including invalid; norm1000 bins, not physical negatives',versions=shapes))
    for path,sha in p.load(root/'qualification.json')['evaluator_sha256'].items():assert p.digest(path)==sha,path
    partitions=p.load(r.ROOT/'cpu-03/evaluator-partitions.json')
    assert p.digest(r.TRUTH)==partitions['truth_sha256']
    truth=p.load(r.TRUTH)
    scored={k:r.assess_outputs(truth,partitions['hidden10'],v) for k,v in frozen.items()}
    p.write(output/'offline-results.json',r.family_outcomes(scored,[]))


def geometry_diagnostic_rows(logits, positions, rows, record, coordinate_ids):
    """Detached compact logits only: no model backward or cached-path parity claim."""
    import torch
    errors={x[0] for x in erroneous_slots(rows)};result=[];n=len(record['prompt_token_ids'])
    expected=tuple(sorted({n+j-1 for row in rows for j in row['coordinate_positions']}))
    assert tuple(positions)==expected
    for row in rows:
        legal={pos:(lo,hi) for pos,lo,hi in legal_slots(row)}
        for slot,pos in enumerate(row['coordinate_positions']):
            z=logits[0,positions.index(n+pos-1)].detach().float().clone().requires_grad_()
            emitted=record['token_ids'][pos];assert emitted==coordinate_ids[row['bbox'][slot]]
            top=z.max();base=dict(order=row['order'],slot=slot,generated_position=pos,causal_position=n+pos-1,
                emitted_bin=row['bbox'][slot],emitted_token_id=emitted,emitted_rank=1+int((z>z[emitted]).sum()),
                replay_argmax_token=int(z.argmax()),argmax_ties=int((z==top).sum()),
                emitted_is_argmax=bool(z[emitted]==top),emitted_equals_replay_argmax=emitted==int(z.argmax()))
            if pos not in legal:
                result.append(dict(base,legal_empty=True,eligible_error=False,reason='own_start999; charged at earlier start'));continue
            lo,hi=legal[pos];ids=list(coordinate_ids[lo:hi]);mask=torch.ones(len(z),dtype=torch.bool,device=z.device);mask[ids]=False
            old=torch.logsumexp(z,0)-torch.logsumexp(z[ids],0);new=max_geometry_margin(z,ids)
            go,=torch.autograd.grad(old,z,retain_graph=True);gn,=torch.autograd.grad(new,z)
            eligible=pos in errors;old_scale=1/(len(rows)*len(legal));new_scale=.1/len(errors) if eligible else 0.
            def measures(g,scale):
                g=g*scale
                return dict(l2=float(g.norm()),linf=float(g.abs().max()),nonzero=int(torch.count_nonzero(g)),emitted=float(g[emitted]),
                            legal_sum=float(g[ids].sum()),illegal_sum=float(g[mask].sum()))
            ml=z[ids].max();mi=z[mask].max()
            result.append(dict(base,legal_empty=False,legal_range=[lo,hi],emitted_legal=lo<=row['bbox'][slot]<hi,eligible_error=eligible,
                legal_mass=float(torch.exp(-old.detach())),max_legal=float(ml.detach()),max_illegal=float(mi.detach()),illegal_minus_legal=float((mi-ml).detach()),
                legal_max_ties=int((z[ids]==ml).sum()),illegal_max_ties=int((z[mask]==mi).sum()),
                old_raw=measures(go,1),new_raw=measures(gn,1),old_image=measures(go,old_scale),new_weighted_image=measures(gn,new_scale),
                old_equal18=measures(go,old_scale/18),new_equal18=measures(gn,new_scale/18),
                old_rank_backward=measures(go,old_scale*8/18),new_rank_backward=measures(gn,new_scale*8/18)))
    return result


def geometry_replay(output, root, updates=1, preservation_weight=0, preservation_bank_sha256=None,
                    geometry_weight=0, start_checkpoint=None, recipe_sha256=None):
    import time,torch
    import torch.distributed as dist
    from src.qwen.native import exact_history_inputs
    from src.losses.vocab import build_token_vocabulary_groups
    from src.artifacts.git_identity import verify_source_identity
    assert preservation_weight==0 and preservation_bank_sha256 is None
    binding=geometry_binding(root,start_checkpoint,geometry_weight,recipe_sha256);assert binding is not None
    rank,out,sources,source=start(output,root);begin=time.monotonic()
    spec=p.load(root/'qualification.json')['geometry'];plan=p.load(spec['diagnostic_plan'])
    assert p.digest(spec['diagnostic_plan'])==spec['diagnostic_plan_sha256']
    inputs={x['image_id']:x for x in p.load(INPUTS)}
    records={x['image_id']:x for x in frozen_records(Path(plan['rollout']),list(inputs))}
    assert p.digest(Path(plan['rollout'])/'frozen.json')==plan['frozen_sha256']
    dist.init_process_group('nccl')
    q,delta,composition=p.compose(start_checkpoint,evaluation=False);q.model.eval()
    p.write(out/'composition.json',composition);p.write(out/'geometry.json',binding)
    fingerprint=parameter_identity(q.model);p.write(out/'producer.json',dict(parameters=fingerprint,sha256=identity(fingerprint)))
    vocab=build_token_vocabulary_groups(q.token_identity,tokenizer=q.tokenizer)
    by={x['image_id']:x for x in plan['images']}
    for i in sorted(inputs)[rank::8]:
        record=records[i];rows,_=observations(record,q.tokenizer)
        expected=by[i];full=record['prompt_token_ids']+record['token_ids']
        positions=tuple(sorted({len(record['prompt_token_ids'])+j-1 for row in rows for j in row['coordinate_positions']}))
        assert identity(full)==expected['input_sha256'] and list(positions)==expected['positions'] and identity(rows)==expected['rows_sha256']
        assert record['raw_identity']==expected['raw_identity']
        batch=native_batch(q,inputs[i]);kwargs=exact_history_inputs(q.model,batch.inputs,[full],pad_token_id=q.tokenizer.pad_token_id)
        kwargs['logits_to_keep']=torch.tensor(positions or (len(record['prompt_token_ids'])-1,),device='cuda')
        before=time.monotonic()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):raw=q.model(**kwargs).logits
        # Clone outside inference_mode makes detached, logit-local differentiation safe.
        logits=raw.detach().cpu().float().clone();del raw
        details=geometry_diagnostic_rows(logits,positions,rows,record,vocab.coordinate)
        p.write(out/f'{i}.json',dict(image_id=i,source_raw_identity=record['raw_identity'],input_sha256=identity(full),positions=list(positions),
            input_tokens=len(full),visual_tokens=int(__import__('math').prod(record['image_grid_thw'])//4),logits_sha256=p.tensor_hash(logits),
            seconds=time.monotonic()-before,rows=details,producer_sha256=identity(fingerprint)))
    assert parameter_identity(q.model)==fingerprint and all(x.grad is None for x in q.model.parameters())
    dist.barrier();verify_source_identity(source,required_paths=sources)
    p.write(out/'complete.json',dict(status='complete',source=source,read_only=True,model_backwards=0,generation_calls=0,
        wall_seconds=time.monotonic()-begin,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),
        artifacts={x.name:p.digest(x) for x in out.glob('*.json')}))
    dist.destroy_process_group()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('run','readback','offline','geometry-replay'))
    parser.add_argument('--root',type=Path,default=ROOT);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--updates',type=int,choices=(1,16,64),default=1)
    parser.add_argument('--preservation-weight',type=float,choices=(0,.25),default=0)
    parser.add_argument('--preservation-bank-sha256')
    parser.add_argument('--geometry-weight',type=float,choices=(0,.1),default=0)
    parser.add_argument('--start-checkpoint',type=Path)
    parser.add_argument('--recipe-sha256')
    a=parser.parse_args();{'run':run,'readback':readback,'offline':offline,'geometry-replay':geometry_replay}[a.command](
        a.output,a.root,a.updates,a.preservation_weight,a.preservation_bank_sha256,a.geometry_weight,a.start_checkpoint,a.recipe_sha256)


if __name__=='__main__':main()
