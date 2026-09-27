"""CPU real-prefix row credit; research unit owns releases and scientific policy."""
from __future__ import annotations

import argparse
from pathlib import Path

from probes import iterative_positive as p
from probes.hidden_human_recovery import negative_evidence
from src.eval.saved_rows import one_to_one_matches
from src.inference.parsing import parse_compact_object_box_closed

ROOT = p.ROOT.parent / 'rollout-row-credit-01'
TRUTH = p.OLD / 'preparation-v3/evaluator/truth.json'
ZERO = p.ROOT / 'round-01/evaluation-zero'
RELEASE_SHA = '1445ef1f39673785477729e2da439bee124b38eef0e0ee4131b43c9fde305668'


def split_reference(truth, percent):
    """Trusted preparation only. Consumer gets the returned retained view alone."""
    total = sum(len(r['objects']) for r in truth)
    quotas = {r['image_id']: len(r['objects'])*percent//100 for r in truth}
    remainder = total*percent//100-sum(quotas.values())
    for r in sorted(truth, key=lambda r: (-(len(r['objects'])*percent % 100), r['image_id']))[:remainder]:
        quotas[r['image_id']] += 1
    visible, hidden = [], []
    for r in sorted(truth, key=lambda r:r['image_id']):
        ordered = sorted(r['objects'], key=lambda o:p.stable(f"row-credit-v1:92711:{r['image_id']}:{o['coco_ann_id']}"))
        ids = {o['coco_ann_id'] for o in ordered[:quotas[r['image_id']]]}
        assert len({o['coco_ann_id'] for o in ordered}) == len(ordered)
        # Whitelist: no inherited hidden bank, counts, categories or coordinates.
        row = {k:r[k] for k in ('image_id','cohort','image_path','image_sha256','width','height')}
        visible.append(dict(row, objects=[o for o in r['objects'] if o['coco_ann_id'] not in ids]))
        hidden.extend([r['image_id'],i] for i in sorted(ids))
    return visible, hidden


def frontend():
    from src.qwen.runtime_loading import QwenLoadOptions, load_qwen_components_from_options
    return load_qwen_components_from_options(QwenLoadOptions(p.load(p.POLICY)['base_model'],'fp32','sdpa',load_model=False))


def parse(record):
    return parse_compact_object_box_closed(record['text'],row_id=record['request_id'],row_index=0,
                                          image_width=record['width'],image_height=record['height'])


def aligned_tokens(record, tokenizer):
    encoded = tokenizer(record['text'], add_special_tokens=False, return_offsets_mapping=True)
    if list(encoded['input_ids']) != record['token_ids'] or tokenizer.decode(record['token_ids'],skip_special_tokens=False) != record['text']:
        raise ValueError('saved token/text alignment differs')
    return encoded


def row_positions(row, encoded):
    a,z = row['char_start'],row['char_end']
    positions = [i for i,(s,e) in enumerate(encoded['offset_mapping']) if s>=a and e<=z and e>s]
    if not positions or encoded['offset_mapping'][positions[0]][0]!=a or encoded['offset_mapping'][positions[-1]][1]!=z:
        raise ValueError('row boundary/token alignment differs')
    if positions != list(range(positions[0],positions[-1]+1)):
        raise ValueError('noncontiguous row tokens')
    return positions


def render_row(image, obj):
    from src.data.examples import RawExample, RawObject, ImageRef, SourceProvenance
    from src.config.models import TemplateConfig
    from src.templates.renderer import render_example
    raw = RawExample(str(image['image_id']),ImageRef(image['image_path'],Path(image['image_path']),image['width'],image['height'],{}),
                     (RawObject('row',obj['desc'],tuple(obj['bbox_2d']),{}),),{},SourceProvenance(Path(__file__),1,'cpu','retained_or_prediction'))
    return render_example(raw,TemplateConfig(object_field_order='desc_first',object_ordering='geo_sorted_xy',
                                            assistant_format='object_box_closed',prompt=p.load(p.POLICY)['prompt']))


def credit_plan(visible, records, tokenizer):
    """Prediction + retained labels only; no truth path or prior proposal bank."""
    result=[]
    assert {r['image_id'] for r in records} == {r['image_id'] for r in visible}
    for image in visible:
        r=next(r for r in records if r['image_id']==image['image_id'])
        assert r['crop']==[0,0,image['width'],image['height']]
        encoded=aligned_tokens(r,tokenizer); parsed=parse(r)
        valid,_=p.candidates([r]); by={o['generated_order']:o for o in parsed.predictions}
        refs=[dict(owner_id=str(o['coco_ann_id']),reference_coord_bins_1000=o['bbox_2d']) for o in image['objects']]
        matches=one_to_one_matches(refs,valid,.5)
        positives=[]; covered=set(); disagreements=[]
        negative=negative_evidence([r],tokenizer)
        complete_rows=list(parsed.predictions)
        for n in negative['complete_geometry_invalid']:
            d=n['parser_drop']; raw=d['raw_text']
            description=raw.split('<|object_ref_start|>',1)[1].split('<|object_ref_end|>',1)[0]
            complete_rows.append(dict(d,description=description,coord_bins=n['coordinate_bins']))
        first={}; repeated={}; occurrences=[]
        for o in sorted(complete_rows,key=lambda o:o['generated_order']):
            key=p.canonical([o['description'],o['coord_bins']])
            positions=row_positions(o,encoded)
            entry=dict(order=o['generated_order'],key=key,positions=positions,description=o['description'],bbox=o['coord_bins'])
            occurrences.append(entry)
            if key in first:
                repeated.setdefault(key,entry)
            else: first[key]=entry
        for m in matches:
            o=image['objects'][m['reference_index']]
            order=int(m['prediction_id'].rsplit(':p',1)[1]); row=by[order]
            if row['description']!=o['desc']:
                disagreements.append(m); continue
            covered.add(o['coco_ann_id'])
            # Keep annotation coverage, but a repeated literal never gets competing positive credit.
            key=p.canonical([row['description'],row['coord_bins']])
            if first[key]['order']==order:
                positives.append(dict(first[key],annotation_id=o['coco_ann_id']))
        fns=sorted((o for o in image['objects'] if o['coco_ann_id'] not in covered),
                   key=lambda o:p.stable(f"row-credit-fn-v1:{image['image_id']}:{o['coco_ann_id']}"))
        eos=tokenizer.convert_tokens_to_ids('<|im_end|>')
        ends=[i for i,t in enumerate(r['token_ids']) if t==eos]
        # Actual termination only. Parser-valid completion required; no guessed cap closure.
        terminal=(r['stop_reason'] in ('im_end','eos') and ends==[len(r['token_ids'])-1]
                  and not negative['malformed_or_censored'] and (not r['token_ids'][:-1] or r['token_ids'][-2]==tokenizer.convert_tokens_to_ids('<|box_end|>')))
        fn=None
        if fns and terminal:
            rendered=render_row(image,fns[0]); text=rendered.assistant_content_text
            ids=tokenizer.encode(text,add_special_tokens=False)
            fn=dict(annotation_id=fns[0]['coco_ann_id'],description=fns[0]['desc'],bbox=fns[0]['bbox_2d'],
                    token_ids=ids,positions=list(range(len(r['token_ids'])-1,len(r['token_ids'])-1+len(ids))))
        unique={}
        for n in negative['complete_geometry_invalid']:
            unique.setdefault(n['literal_repeat_key'],n)
        result.append(dict(image_id=image['image_id'],cohort=image['cohort'],request_id=r['request_id'],
            M=positives,F=fn,D=list(repeated.values()),G=list(unique.values()),observations=occurrences,
            known_fn_ids=[o['coco_ann_id'] for o in fns],fn_ineligible_reason=None if terminal else 'capped_or_ambiguous_termination',
            category_disagreements=disagreements,negative_evidence=negative,
            matched_literal_veto=len(covered)-len(positives),retained_count=len(image['objects'])))
    return result


def insertion_plan(plans, records):
    """Move only the frozen F row to the first greater observed anchor."""
    records={x['image_id']:x for x in records};result=[]
    for plan in plans:
        fn=plan['F']
        if fn is None:
            result.append(dict(plan));continue
        successor=next(o for o in sorted(plan['observations'],key=lambda o:o['order'])
                       if tuple(o['bbox'][:2])>tuple(fn['bbox'][:2]))
        cut=successor['positions'][0];record=records[plan['image_id']]
        assert record['token_ids'][cut]==fn['token_ids'][0]
        assert 0<=cut<len(record['token_ids'])-1
        moved=dict(fn,prefix_cut=cut,positions=list(range(cut,cut+len(fn['token_ids']))))
        result.append(dict(plan,F=moved,insertion=dict(cut=cut,terminal_cut=len(record['token_ids'])-1,
            successor_order=successor['order'],successor_bbox=successor['bbox'],
            successor_M_credit=any(o['positions'][0]==cut for o in plan['M']))))
    return result


def positive_sequence(image, record, row, tokenizer, fn=False):
    """Render atom metadata only; saved causal IDs remain authoritative for M."""
    from src.packing.planner import PackedSegment, PackedSequence
    from src.supervision.tokens import TokenAtom, build_token_sequence_from_packed_supervision
    rendered=render_row(image,dict(desc=row['description'],bbox_2d=row['bbox']))
    text=rendered.assistant_content_text
    tokenized=tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
    ids=list(tokenized['input_ids']); positions=row['positions']
    history=list(record['token_ids'])
    if fn:
        assert history[-1]==tokenizer.convert_tokens_to_ids('<|im_end|>')
        cut=row.get('prefix_cut',len(history)-1)
        assert 0<=cut<=len(history)-1
        history=history[:cut]+row['token_ids']
    if [history[i] for i in positions]!=ids:
        raise ValueError('positive rendered metadata differs from saved row tokens')
    prompt=record['prompt_token_ids']; full=tuple(prompt+history)
    atoms=[]
    kinds={'schema_token':'schema','description':'desc_text','coordinate_token':'coordinate'}
    for j,(a,z) in enumerate(tokenized['offset_mapping']):
        spans=[s for s in rendered.spans if s.kind in kinds and s.char_start<=a and z<=s.char_end]
        if len(spans)!=1: raise ValueError('positive atom span alignment differs')
        s=spans[0]; pos=len(prompt)+positions[j]
        atoms.append(TokenAtom(0,0,0,str(image['image_id']),pos,ids[j],kinds[s.kind],s.text,pos,
                               object_id='row',field=s.field,source='actual_prefix',coordinate_target=s.coordinate_target))
    pack=PackedSequence(0,full,(PackedSegment(0,0,0,str(image['image_id']),0,len(full)),),p.MAX_LENGTH)
    return build_token_sequence_from_packed_supervision(pack,atoms)


def retained_sequence(image, q):
    """Coherent retained-label teacher forcing, never a generated-history prefix."""
    from src.data.examples import RawExample, RawObject, ImageRef, SourceProvenance
    objects=sorted(image['objects'],key=lambda o:(o['bbox_2d'][0],o['bbox_2d'][1],o['coco_ann_id']))
    row_ids=tuple('retained:'+str(o['coco_ann_id']) for o in objects)
    assert len(set(row_ids))==len(objects)
    raw=RawExample(str(image['image_id']),ImageRef(image['image_path'],Path(image['image_path']),image['width'],image['height'],{}),
        tuple(RawObject(oid,o['desc'],tuple(o['bbox_2d']),{}) for oid,o in zip(row_ids,objects,strict=True)),{},
        SourceProvenance(ROOT/'cpu-04/retained-10.json',1,'frozen_retained90','rendered_retained_labels'))
    encoded,sequence=p.encode(raw,set(row_ids),q)
    assert tuple(dict.fromkeys(a.object_id for a in sequence.atoms))==row_ids
    return encoded,sequence,row_ids


def retained_objective(logits, sequence, row_ids, vocab, positions):
    """Existing within-row loss, then row mean for this one original image."""
    import torch
    from src.packing.planner import PackedSequence
    from src.supervision.tokens import build_token_sequence_from_packed_supervision
    assert len(set(row_ids))==len(row_ids)
    assert tuple(dict.fromkeys(a.object_id for a in sequence.atoms))==tuple(row_ids)
    assert all(a.token_type in ('schema','desc_text','coordinate') and a.object_id in row_ids for a in sequence.atoms)
    pack=PackedSequence(sequence.pack_index,sequence.input_ids,sequence.segments,p.MAX_LENGTH)
    losses=[];details=[]
    for oid in row_ids:
        atoms=tuple(a for a in sequence.atoms if a.object_id==oid)
        assert {a.field for a in atoms}>={'object_ref_start','description','object_ref_end','box_start','box_end','bbox[0]','bbox[1]','bbox[2]','bbox[3]'}
        row=build_token_sequence_from_packed_supervision(pack,atoms)
        loss,detail=p.image_loss(logits,row,vocab,positions);losses.append(loss)
        details.append(dict(object_id=oid,atoms=len(atoms),loss=float(loss.detach()),**detail))
    return torch.stack(losses).mean() if losses else logits.sum()*0,details


def forward_retained(q, model, image, record, expected, vocab):
    import torch
    from src.qwen.native import exact_history_inputs
    encoded,sequence,row_ids=retained_sequence(image,q)
    positions=tuple(a.causal_logits_position for a in sequence.atoms)
    assert list(sequence.input_ids)==expected['input_ids'] and list(row_ids)==expected['row_ids']
    assert [a.to_artifact_dict() for a in sequence.atoms]==expected['atoms']
    assert list(positions)==expected['positions']
    batch=p.native_request(record,p.load(p.POLICY),q.processor)
    assert list(batch.prompt_token_ids[0])==record['prompt_token_ids']
    assert list(sequence.input_ids[:len(record['prompt_token_ids'])])==record['prompt_token_ids']
    assert list(batch.image_grids[0])==record['image_grid_thw']==list(encoded.image_encoding.image_grid_thw)
    assert batch.media_sha256[0]==record['media_sha256']
    kwargs=exact_history_inputs(q.model,batch.inputs,[sequence.input_ids],pad_token_id=q.tokenizer.pad_token_id)
    kwargs['logits_to_keep']=torch.tensor(positions,device='cuda')
    with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(**kwargs).logits
    loss,rows=retained_objective(logits,sequence,row_ids,vocab,positions)
    return loss,dict(image_id=image['image_id'],branch='S',prefix_source='rendered_retained_labels',row_count=len(row_ids),
        tokens=len(sequence.input_ids),input_sha256=p.stable(p.canonical(sequence.input_ids)),positions=list(positions),
        atoms_sha256=p.stable(p.canonical(expected['atoms'])),logits_sha256=p.tensor_hash(logits),loss=float(loss.detach()),row_losses=rows)


def selected_logits(logits, positions, targets):
    lookup={v:i for i,v in enumerate(positions)}
    if len(lookup)!=len(positions): raise ValueError('duplicate logit positions')
    return logits[0,[lookup[t-1] for t in targets]].float()


def row_unlikelihood(logits, positions, full_ids, targets):
    import torch
    rows=selected_logits(logits,positions,targets)
    labels=torch.tensor([full_ids[t] for t in targets],device=rows.device)
    target=rows.gather(1,labels[:,None]).squeeze(1)
    rest=rows.scatter(1,labels[:,None],-torch.inf).logsumexp(-1)
    token_logp=-torch.nn.functional.softplus(rest-target)
    logp=token_logp.sum()
    if logp > -0.6931471805599453:
        # Disjoint first-failure events retain the complement even when p rounds to1.
        log_failure=-torch.nn.functional.softplus(target-rest)
        prefix=torch.cat((token_logp.new_zeros(1),token_logp.cumsum(0)[:-1]))
        loss=-torch.logsumexp(prefix+log_failure,0)
    else:
        # Small row probabilities must not gain cancellation noise from 1-P.
        loss=-torch.log1p(-torch.exp(logp))
    return loss,logp.exp()


def geometry_loss(logits, positions, evidence, prompt_length, coordinate_ids):
    """Conditional illegal mass at certified own-prefix slots; invalids are not GT."""
    import torch
    slots=[]
    for j in evidence['illegal_slots']:
        predecessor=evidence['coordinate_bins'][j-2]
        slots.append((j-2,998) if predecessor==999 else (j,predecessor))
    terms=[]
    for slot,threshold in slots:
        target=prompt_length+evidence['coordinate_token_positions'][slot]
        row=selected_logits(logits,positions,[target])[0]
        coord=row[list(coordinate_ids)]
        # At an impossible-start slot only bin999 is illegal; end slots require > predecessor.
        legal=coord[:999] if evidence['coordinate_bins'][slot]==999 and slot in (0,1) else coord[threshold+1:]
        assert legal.numel()>0
        mass=torch.logsumexp(coord,0)-torch.logsumexp(legal,0)
        gate=torch.logsumexp(row,0)-torch.logsumexp(coord,0)
        terms.append(mass+.1*gate)
    return torch.stack(terms).mean()


def image_objective(arm, plan, record, image, tokenizer, vocab, logits, positions, fn_logits=None, fn_positions=None):
    """Per-row branch mean, then M+F+.1D+.01G; caller averages original images."""
    import torch
    if arm not in ('A','B','C'): raise ValueError('unknown arm')
    zero=logits.sum()*0
    def mean(values): return torch.stack(values).mean() if values else zero
    m=[p.image_loss(logits,positive_sequence(image,record,row,tokenizer),vocab,positions)[0] for row in plan['M']]
    f=[];d=[];g=[];probabilities=[]
    if arm!='A' and plan['F'] is not None:
        if fn_logits is None: raise ValueError('FN needs its actual appended continuation forward')
        f=[p.image_loss(fn_logits,positive_sequence(image,record,plan['F'],tokenizer,True),vocab,fn_positions)[0]]
    if arm=='C':
        full=record['prompt_token_ids']+record['token_ids']; n=len(record['prompt_token_ids'])
        for row in plan['D']:
            loss,prob=row_unlikelihood(logits,positions,full,[n+i for i in row['positions']]); d.append(loss); probabilities.append(float(prob.detach()))
        g=[geometry_loss(logits,positions,e,n,vocab.coordinate) for e in plan['G']]
    branches=dict(M=mean(m),F=mean(f),D=mean(d),G=mean(g))
    return branches['M']+branches['F']+.1*branches['D']+.01*branches['G'],dict(branches=branches,row_probabilities=probabilities)


def objective_positions(plan, record, arm):
    n=len(record['prompt_token_ids']); targets=set()
    for row in plan['M']: targets.update(n+i for i in row['positions'])
    if arm=='C':
        for row in plan['D']: targets.update(n+i for i in row['positions'])
        for e in plan['G']:
            for j in e['illegal_slots']:
                slot=j-2 if e['coordinate_bins'][j-2]==999 else j
                targets.add(n+e['coordinate_token_positions'][slot])
    return tuple(sorted(t-1 for t in targets)) or (n-1,)



def forward_credit(q, model, plan, record, image, vocab, arm, fn=False, bookkeeping=False, diagnostics=False):
    """Native exact-prefix seam for the later released slice; not called by CPU prepare."""
    import torch
    from src.qwen.native import exact_history_inputs
    batch=p.native_request(record,p.load(p.POLICY),q.processor)
    assert list(batch.prompt_token_ids[0])==record['prompt_token_ids']
    assert list(batch.image_grids[0])==record['image_grid_thw']
    assert batch.media_sha256[0]==record['media_sha256']
    if fn and plan['F'] is not None:
        sequence=positive_sequence(image,record,plan['F'],q.tokenizer,True)
        full=sequence.input_ids
        positions=tuple(a.causal_logits_position for a in sequence.atoms)
    else:
        full=record['prompt_token_ids']+record['token_ids']
        positions=objective_positions(plan,record,arm)
    kwargs=exact_history_inputs(q.model,batch.inputs,[full],pad_token_id=q.tokenizer.pad_token_id)
    kwargs['logits_to_keep']=torch.tensor(positions,device='cuda')
    with torch.autocast('cuda',dtype=torch.bfloat16):
        logits=model(**kwargs).logits
    if fn:
        loss=p.image_loss(logits,sequence,vocab,positions)[0] if plan['F'] is not None else logits.sum()*0
        zero=logits.sum()*0
        detail=dict(branches=dict(M=zero,F=loss,D=zero,G=zero))
    else:
        loss,detail=image_objective(arm,dict(plan,F=None),record,image,q.tokenizer,vocab,logits,positions)
    evidence=dict(image_id=image['image_id'],branch='F' if fn else 'M_D_G',tokens=len(full),
                  input_sha256=p.stable(p.canonical(list(full))),positions=list(positions),
                  logits_sha256=p.tensor_hash(logits),loss=float(loss.detach()),
                  row_probabilities=detail.get('row_probabilities',[]))
    if bookkeeping:
        counts=dict(M=0 if fn else len(plan['M']),F=int(fn and plan['F'] is not None),
                    D=len(plan['D']) if not fn and arm=='C' else 0,G=len(plan['G']) if not fn and arm=='C' else 0)
        evidence.update(branch_scalars={k:float(v.detach()) for k,v in detail['branches'].items()},branch_counts=counts)
        if diagnostics: evidence['logit_diagnostics']=logit_diagnostics(detail['branches'],counts,logits)
    return loss,evidence


SLICE_IMAGES=(1584,2299,2685,4134,7511,14038,351017,477415)


def slice_run(output, cpu_root, arm, reload_from=None):
    """Proposed one-update/fresh-reload entry. Requires a separate lead runtime release."""
    import os
    import time
    import torch
    import torch.distributed as dist
    from contextlib import nullcontext
    from torch.nn.parallel import DistributedDataParallel
    from src.losses.vocab import build_token_vocabulary_groups
    from src.artifacts.git_identity import capture_source_identity,verify_source_identity
    rank=int(os.environ['RANK']); assert int(os.environ['WORLD_SIZE'])==8
    qualifier=p.load(cpu_root/'qualification.json')
    for path,sha in qualifier['sha256'].items(): assert p.digest(path)==sha,path
    sources=['probes/rollout_row_credit.py','probes/iterative_positive.py','probes/hidden_human_recovery.py',
             *sorted(str(x) for x in Path('src').rglob('*.py'))]
    identity=capture_source_identity(sources)
    out=output/f'rank-{rank}';out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    p.write(out/'entry.json',dict(pid=os.getpid(),rank=rank,source=identity,start=time.time()))
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']));torch.manual_seed(92711)
    q,delta,composition=p.compose(reload_from/'checkpoint-1' if reload_from else Path(p.load(p.POLICY)['checkpoint']))
    p.write(out/'composition.json',composition)
    vocab=build_token_vocabulary_groups(q.token_identity,tokenizer=q.tokenizer)
    image=next(i for i in p.load(cpu_root/'retained-10.json') if i['image_id']==SLICE_IMAGES[rank])
    plan=next(i for i in p.load(cpu_root/'credit-plan.json') if i['image_id']==image['image_id'])
    record=next(i for i in p.read_evaluation(ZERO) if i['image_id']==image['image_id'])
    if not reload_from:
        dist.init_process_group('nccl')
        params=[x for x in q.model.parameters() if x.requires_grad]
        delta_ids={id(x) for x in delta.delta_tensors().values()}
        optimizer=torch.optim.AdamW([dict(params=[x for x in params if id(x) not in delta_ids],lr=1e-5),
                                    dict(params=list(delta.delta_tensors().values()),lr=5e-6)],betas=(.9,.999),eps=1e-8,weight_decay=0)
        q.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});q.model.enable_input_require_grads()
        q.model.train();model=DistributedDataParallel(q.model,device_ids=[int(os.environ['LOCAL_RANK'])],broadcast_buffers=False)
        if rank==0: p.save_checkpoint(q,delta,output/'checkpoint-0')
        dist.barrier();optimizer.zero_grad(set_to_none=True)
        branches=[False] if arm=='A' else [False,True]
        for j,fn in enumerate(branches):
            with model.no_sync() if j+1<len(branches) else nullcontext():
                loss,evidence=forward_credit(q,model,plan,record,image,vocab,arm,fn)
                assert torch.isfinite(loss);loss.backward()
            p.write(out/f'micro-{j}.json',evidence)
        norms={n:float(x.grad.float().norm()) if x.grad is not None else None for n,x in q.model.named_parameters() if x.requires_grad}
        assert all(v is not None and __import__('math').isfinite(v) for v in norms.values())
        assert all(any(v>0 for n,v in norms.items() if tag in n) for tag in ('lora_','embed_tokens.shared_embed_delta','lm_head.shared_embed_delta'))
        torch.nn.utils.clip_grad_norm_(params,1,error_if_nonfinite=True);optimizer.step()
        assert all(torch.isfinite(x).all() for x in params)
        p.write(out/'gradients.json',norms)
        if rank==0: p.save_checkpoint(q,delta,output/'checkpoint-1')
        dist.barrier()
    q.model.eval()
    with torch.no_grad():
        readback=[forward_credit(q,q.model,plan,record,image,vocab,arm,fn)[1] for fn in ([False] if arm=='A' else [False,True])]
    if reload_from: assert readback==p.load(reload_from/f'rank-{rank}/final-forward.json'),'fresh reload differs'
    p.write(out/'final-forward.json',readback)
    verify_source_identity(identity,required_paths=sources)
    p.write(out/'complete.json',dict(status='complete',arm=arm,reloaded=bool(reload_from),source=identity,
             wall_seconds=time.monotonic()-start,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved()))
    if not reload_from: dist.destroy_process_group()


FAMILY = ROOT/'family-01'


def family_jobs(image_ids, arm, rank):
    if arm not in ('A','B','C','I','S') or len(image_ids)!=18 or len(set(image_ids))!=18 or not 0<=rank<8:
        raise ValueError('family requires A/B/C/I/S, exactly18images and rank0..7')
    jobs=[dict(image_id=i,fn=fn,weight=8/18) for i in sorted(image_ids)[rank::8]
          for fn in ([False] if arm in ('A','S') else [False,True])]
    return [dict(j,sync=k==len(jobs)-1) for k,j in enumerate(jobs)]


def accumulate_family_step(model, jobs, forward):
    """Same uneven-rank backward consumer for CPU falsification and real training."""
    from contextlib import nullcontext
    import torch
    evidence=[]
    for job in jobs:
        with nullcontext() if job['sync'] else model.no_sync():
            loss,row=forward(job)
            assert torch.isfinite(loss)
            (loss*job['weight']).backward()
        evidence.append(dict(row,image_weight=job['weight'],sync=job['sync']))
    return evidence


def logit_diagnostics(branches, counts, logits):
    """Stop autograd at existing logits; no parameter diagnostic backward or tensors saved."""
    import torch
    result={}
    for name,loss in branches.items():
        coefficient={'M':1.,'F':1.,'D':.1,'G':.01}[name]
        def measure(grad):
            grad=grad.detach().float()
            assert torch.isfinite(grad).all()
            return dict(l1=float(grad.abs().sum()),l2=float(grad.norm()),linf=float(grad.abs().max()),
                        support_rows=int((grad.abs().sum(-1)>0).sum()),support_elements=int(torch.count_nonzero(grad)))
        if counts[name]:
            grad,=torch.autograd.grad(loss,logits,retain_graph=True)
            norms=measure(grad)
            if coefficient==1: weighted=norms.copy()
            else:
                weighted_grad,=torch.autograd.grad(coefficient*loss,logits,retain_graph=True)
                weighted=measure(weighted_grad)
        else:
            norms=dict(l1=0.,l2=0.,linf=0.,support_rows=0,support_elements=0);weighted=norms.copy()
        result[name]=dict(count=counts[name],coefficient=coefficient,unweighted=norms,weighted=weighted,
                          support_rows=norms['support_rows'],support_elements=norms['support_elements'])
    return result


def family_data():
    cpu=ROOT/'cpu-04'
    return ({x['image_id']:x for x in p.load(cpu/'retained-10.json')},
            {x['image_id']:x for x in p.load(cpu/'credit-plan.json')},
            {x['image_id']:x for x in p.read_evaluation(ZERO)})


def family_start(output, family_root):
    import os,time,torch
    from src.artifacts.git_identity import capture_source_identity
    qualifier=p.load(family_root/'qualification.json')
    for path,sha in qualifier['sha256'].items(): assert p.digest(path)==sha,path
    rank=int(os.environ['RANK']);assert int(os.environ['WORLD_SIZE'])==8
    sources=['probes/rollout_row_credit.py','probes/iterative_positive.py','probes/hidden_human_recovery.py',
             *sorted(str(x) for x in Path('src').rglob('*.py'))]
    identity=capture_source_identity(sources)
    out=output/f'rank-{rank}';out.mkdir(parents=True,exist_ok=False)
    p.write(out/'entry.json',dict(pid=os.getpid(),rank=rank,source=identity,start=time.time()))
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']));torch.manual_seed(92711)
    return rank,out,sources,identity


def family_schedule(updates=8):
    if updates not in (8,16):raise ValueError('only frozen eight or sixteen updates')
    return [dict(step=step,save=step in (4,8,16),diagnostics=step in (1,8)) for step in range(1,updates+1)]


def checkpoint_gate(checkpoint, reference, bindings):
    """CPU exact tensor equality, with immutable control and input bindings."""
    import torch
    from safetensors.torch import load_file
    for path,sha in bindings.items():assert p.digest(path)==sha,path
    identities=[p.load(root/'identity.json') for root in (checkpoint,reference)]
    assert set(identities[0])==set(identities[1])
    tensors=0
    for name in identities[0]:
        for root,identity in zip((checkpoint,reference),identities):assert p.digest(root/name)==identity[name]
        if name.endswith('.safetensors'):
            a,b=load_file(str(checkpoint/name)),load_file(str(reference/name))
            assert a.keys()==b.keys()
            for key in a:
                assert a[key].dtype==b[key].dtype and torch.equal(a[key],b[key]),(name,key)
                tensors+=1
        elif name=='adapter/adapter_config.json':
            configs=[p.load(root/name) for root in (checkpoint,reference)]
            for config in configs:
                assert isinstance(config,dict),name
                modules=config.get('target_modules')
                assert isinstance(modules,list) and all(isinstance(x,str) for x in modules),name
                config['target_modules']=sorted(modules)
            assert configs[0]==configs[1],name
        else:assert identities[0][name]==identities[1][name],name
    assert tensors>0
    return dict(status='exact',tensors=tensors,checkpoint=str(checkpoint),reference=str(reference),bindings=bindings)


def tail_prefix(record, tokenizer, rule):
    """Fixed prediction-only counterfactual; actual stop facts stay unchanged."""
    assert rule in ('literal_repeat','geometry_invalid')
    encoded=aligned_tokens(record,tokenizer);parsed=parse(record)
    invalid=negative_evidence([record],tokenizer)['complete_geometry_invalid']
    rows=[dict(o,invalid=False) for o in parsed.predictions]
    for n in invalid:
        d=n['parser_drop']
        rows.append(dict(d,invalid=True,description=d['raw_text'].split('<|object_ref_start|>',1)[1].split('<|object_ref_end|>',1)[0],coord_bins=n['coordinate_bins']))
    seen=set();event=None
    for row in sorted(rows,key=lambda x:x['generated_order']):
        key=p.canonical([row['description'],row['coord_bins']])
        if (rule=='geometry_invalid' and row['invalid']) or (rule=='literal_repeat' and key in seen):
            event=dict(order=row['generated_order'],key=key,char_cut=row['char_start'],token_cut=row_positions(row,encoded)[0]);break
        seen.add(key)
    cut=len(record['token_ids']) if event is None else event['token_cut']
    text=record['text'] if event is None else record['text'][:event['char_cut']]
    assert tokenizer.decode(record['token_ids'][:cut],skip_special_tokens=False)==text
    return dict(record,text=text,token_ids=record['token_ids'][:cut],generated_tokens=cut),dict(rule=rule,event=event,
        original_text_sha256=p.stable(record['text']),original_tokens_sha256=p.stable(p.canonical(record['token_ids'])),
        original_stop_reason=record['stop_reason'],original_generated_tokens=len(record['token_ids']),prefix_tokens=cut)


def family_train(output, arm, family_root, updates=8):
    import os,time,math,torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from src.losses.vocab import build_token_vocabulary_groups
    from src.artifacts.git_identity import verify_source_identity
    schedule=family_schedule(updates)
    assert updates==8 or arm=='S'
    rank,out,sources,identity=family_start(output,family_root);start=time.monotonic()
    if arm=='S':
        images={x['image_id']:x for x in p.load(ROOT/'cpu-04/retained-10.json')}
        records={x['image_id']:x for x in p.load(family_root/'inputs.json')}
        encodings={x['image_id']:x for x in p.load(family_root/'encodings.json')}
    else:images,plans,records=family_data()
    jobs=family_jobs(list(images),arm,rank)
    if arm=='I':
        moved=insertion_plan(list(plans.values()),list(records.values()))
        assert moved==p.load(family_root/'credit-plan.json')
        plans={x['image_id']:x for x in moved}
    dist.init_process_group('nccl')
    q,delta,composition=p.compose(Path(p.load(p.POLICY)['checkpoint']))
    load_seconds=time.monotonic()-start;p.write(out/'composition.json',composition)
    vocab=build_token_vocabulary_groups(q.token_identity,tokenizer=q.tokenizer)
    params=[x for x in q.model.parameters() if x.requires_grad];delta_ids={id(x) for x in delta.delta_tensors().values()}
    optimizer=torch.optim.AdamW([dict(params=[x for x in params if id(x) not in delta_ids],lr=1e-5),
                                dict(params=list(delta.delta_tensors().values()),lr=5e-6)],betas=(.9,.999),eps=1e-8,weight_decay=0)
    assert not optimizer.state
    q.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});q.model.enable_input_require_grads()
    q.model.train();model=DistributedDataParallel(q.model,device_ids=[int(os.environ['LOCAL_RANK'])],broadcast_buffers=False)
    if rank==0:p.save_checkpoint(q,delta,output/'checkpoint-0')
    dist.barrier()
    for scheduled in schedule:
        step=scheduled['step']
        optimizer.zero_grad(set_to_none=True);before=time.monotonic()
        def forward(job):
            i=job['image_id']
            if arm=='S':return forward_retained(q,model,images[i],records[i],encodings[i],vocab)
            return forward_credit(q,model,plans[i],records[i],images[i],vocab,'B' if arm=='I' else arm,job['fn'],
                                  bookkeeping=True,diagnostics=scheduled['diagnostics'])
        evidence=accumulate_family_step(model,jobs,forward)
        norms={n:float(x.grad.float().norm()) if x.grad is not None else None for n,x in q.model.named_parameters() if x.requires_grad}
        assert all(v is not None and math.isfinite(v) for v in norms.values())
        assert all(any(v>0 for n,v in norms.items() if tag in n) for tag in ('lora_','embed_tokens.shared_embed_delta','lm_head.shared_embed_delta'))
        synchronized=[None]*8;dist.all_gather_object(synchronized,p.stable(p.canonical(norms)))
        assert len(set(synchronized))==1,'rank gradient norms differ'
        total=float(torch.nn.utils.clip_grad_norm_(params,1,error_if_nonfinite=True));optimizer.step()
        assert all(torch.isfinite(x).all() for x in params)
        torch.cuda.synchronize()
        p.write(out/f'step-{step}.json',dict(step=step,microbatches=evidence,gradient_norms=norms,total_gradient_norm=total,
                 synchronized_norm_hashes=synchronized,lrs=[g['lr'] for g in optimizer.param_groups],seconds=time.monotonic()-before))
        if scheduled['save']:
            if rank==0:p.save_checkpoint(q,delta,output/f'checkpoint-{step}')
            dist.barrier()
    verify_source_identity(identity,required_paths=sources)
    p.write(out/'complete.json',dict(status='complete',arm=arm,steps=updates,source=identity,load_seconds=load_seconds,
        wall_seconds=time.monotonic()-start,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),
        artifacts={x.name:p.digest(x) for x in sorted(out.glob('*.json'))}))
    dist.destroy_process_group()


def family_evaluate(checkpoint, output, family_root):
    import time,torch
    from src.qwen.generation import generate_continuations,NativeGenerationPolicy
    from src.artifacts.git_identity import verify_source_identity
    rank,out,sources,identity=family_start(output,family_root);start=time.monotonic()
    q,delta,composition=p.compose(checkpoint,evaluation=True);q.model.eval()
    load_seconds=time.monotonic()-start;policy=p.load(p.POLICY);records=p.read_evaluation(ZERO);names=[]
    for item in records[rank::8]:
        before=time.monotonic();batch=p.native_request(item,policy,q.processor);prepared=time.monotonic()-before
        assert list(batch.prompt_token_ids[0])==item['prompt_token_ids']
        assert list(batch.image_grids[0])==item['image_grid_thw'] and batch.media_sha256[0]==item['media_sha256']
        before=time.monotonic()
        with torch.inference_mode():
            result=generate_continuations(q.model,batch,extensions=[()],budgets=[3084],eos_token_id=q.tokenizer.convert_tokens_to_ids('<|im_end|>'),pad_token_id=q.tokenizer.pad_token_id,
                 policy=NativeGenerationPolicy(temperature=0,top_p=1,top_k=0,repetition_penalty=1,use_model_defaults=False),seed=None)[0]
        record=dict(item,token_ids=list(result.token_ids),text=q.tokenizer.decode(result.token_ids,skip_special_tokens=False),stop_reason=result.stop_reason,
                    generated_tokens=len(result.token_ids),generation_seconds=time.monotonic()-before,prepare_seconds=prepared,
                    prompt_token_ids=list(batch.prompt_token_ids[0]),image_grid_thw=batch.image_grids[0],media_sha256=batch.media_sha256[0])
        assert len(record['token_ids'])<=3084
        name=str(item['image_id'])+'.json';p.write(out/name,record);names.append(name)
    verify_source_identity(identity,required_paths=sources)
    p.write(out/'complete.json',dict(status='complete',composition=composition,source=identity,load_seconds=load_seconds,
            wall_seconds=time.monotonic()-start,artifacts={name:p.digest(out/name) for name in names}))


def assess_outputs(truth, hidden_keys, records):
    """Offline full-reference assignment plus raw output burdens, without selection feedback."""
    from collections import Counter
    from src.eval.saved_rows import iou_xyxy
    hidden=set(map(tuple,hidden_keys));valid,invalid=p.candidates(records);result=[]
    for image in truth:
        i=image['image_id'];record=next(x for x in records if x['image_id']==i)
        pool=[x for x in valid if x['image_id']==i];drops=[x for x in invalid if x['image_id']==i]
        refs=[dict(owner_id=str(o['coco_ann_id']),reference_coord_bins_1000=o['bbox_2d']) for o in image['objects']]
        matches=one_to_one_matches(refs,pool,.5);by={x['prediction_id']:x for x in pool}
        ids={mode:{kind:[] for kind in ('retained','hidden')} for mode in ('raw','category')}
        denominators={kind:[] for kind in ('retained','hidden')}
        for o in image['objects']:denominators['hidden' if (i,o['coco_ann_id']) in hidden else 'retained'].append(o['coco_ann_id'])
        for match in matches:
            o=image['objects'][match['reference_index']];kind='hidden' if (i,o['coco_ann_id']) in hidden else 'retained'
            ids['raw'][kind].append(o['coco_ann_id'])
            if o['desc']==by[match['prediction_id']]['description']:ids['category'][kind].append(o['coco_ann_id'])
        literal=[p.canonical([x['description'],x['coord_bins_1000']]) for x in pool]
        complete=list(literal)
        for d in drops:
            if d['reason']=='geometry_invalid':
                description=d['raw_text'].split('<|object_ref_start|>',1)[1].split('<|object_ref_end|>',1)[0]
                complete.append(p.canonical([description,[int(s['text'][8:-2]) for s in d['coord_token_spans']]]))
        near=sum(a['description']==b['description'] and a['coord_bins_1000']!=b['coord_bins_1000'] and
                 iou_xyxy(a['coord_bins_1000'],b['coord_bins_1000'])>=.9 for j,a in enumerate(pool) for b in pool[j+1:])
        reasons=Counter(x['reason'] for x in drops)
        result.append(dict(image_id=i,cohort=image['cohort'],ids=ids,denominator_ids=denominators,matches=matches,
            burdens=dict(valid_rows=len(pool),literal_valid_repeats=len(literal)-len(set(literal)),literal_complete_repeats=len(complete)-len(set(complete)),
                         near_repeat_occurrence_pairs=near,geometry_invalid=reasons.get('geometry_invalid',0),
                         malformed=len(drops)-reasons.get('geometry_invalid',0),generated_tokens=record['generated_tokens'],
                         eos=int(record['stop_reason'] in ('im_end','eos')),caps=int(record['stop_reason'] not in ('im_end','eos')),
                         unmatched=len(pool)-len(matches),category_disagreements=len(matches)-sum(map(len,ids['category'].values()))),
            invalid_rows=drops,stop_reason=record['stop_reason']))
    return result


def family_outcomes(scored, selected_fn_keys):
    """All gains/losses use the fixed full570 zero assignment in each declared mode."""
    selected=set(map(tuple,selected_fn_keys));zero={x['image_id']:x for x in scored['zero']};result={}
    for name,images in scored.items():
        rows=[]
        for image in images:
            i=image['image_id'];base=zero[i]
            for mode in ('raw','category'):
                sets={};metrics={}
                for kind in ('retained','hidden'):
                    before=set(base['ids'][mode][kind]);after=set(image['ids'][mode][kind]);universe=set(image['denominator_ids'][kind])
                    sets[kind]=dict(gained=sorted(after-before),lost=sorted(before-after),preserved=sorted(before&after))
                    metrics.update({kind+'_denominator':len(universe),kind+'_incumbent_denominator':len(before),kind+'_FN_denominator':len(universe-before),
                                    kind+'_coverage':len(after),kind+'_FN_acquired':len(after-before),kind+'_incumbent_lost':len(before-after),kind+'_incumbent_preserved':len(before&after)})
                chosen={ann for img,ann in selected if img==i};now=set(image['ids'][mode]['retained'])&chosen;old=set(base['ids'][mode]['retained'])&chosen
                metrics.update(selected_FN_denominator=len(chosen),selected_FN_coverage=len(now),selected_FN_acquired=len(now-old),selected_FN_lost=len(old-now),
                               retained_utility=metrics['retained_FN_acquired']-metrics['retained_incumbent_lost'],
                               total_coverage=metrics['retained_coverage']+metrics['hidden_coverage'])
                rows.append(dict(image_id=i,cohort=image['cohort'],mode=mode,sets=sets,selected_FN_covered=sorted(now),
                                 selected_FN_gained=sorted(now-old),selected_FN_lost=sorted(old-now),metrics=metrics,burdens=image['burdens']))
        summary={}
        for mode in ('raw','category'):
            summary[mode]={}
            for cohort in ('combined','human13','refined5'):
                group=[x for x in rows if x['mode']==mode and (cohort=='combined' or x['cohort']==cohort)]
                summary[mode][cohort]={part:{k:sum(x[part][k] for x in group) for k in rows[0][part]} for part in ('metrics','burdens')}
        result[name]=dict(images=rows,summary=summary)
    return result


def family_offline(family_root, reference_root=None):
    roots={'zero':ZERO,**{f'{arm}-{step}':family_root/f'evaluation-{arm}-{step}' for arm in 'ABC' for step in (4,8)}}
    pairs=(('B','A'),('C','B'))
    if reference_root is not None:
        reuse=p.load(family_root/'control-reuse.json')
        assert str(reference_root)==reuse['reference_root']
        for path,sha in reuse['sha256'].items():assert p.digest(path)==sha,path
        roots={'zero':ZERO,**{f'{arm}-{step}':reference_root/f'evaluation-{arm}-{step}' for arm in 'AB' for step in (4,8)},
               **{f'I-{step}':family_root/f'evaluation-I-{step}' for step in (4,8)}}
        pairs=(('I','A'),('I','B'))
        if reuse.get('arm')=='S':
            roots={'zero':ZERO,**{f'A-{step}':reference_root/f'evaluation-A-{step}' for step in (4,8)},
                   **{f'I-{step}':Path(reuse['insertion_root'])/f'evaluation-I-{step}' for step in (4,8)},
                   **{f'S-{step}':family_root/f'evaluation-S-{step}' for step in (4,8)}}
            pairs=(('S','zero'),('S','A'),('S','I'))
    if reference_root is not None and reuse.get('dose_updates')==16:
        roots={'zero':ZERO,'S-8':Path(reuse['retained_root'])/'evaluation-S-8','S-16':family_root/'evaluation-S-16'}
    # Readback freezes every shard set BEFORE opening any truth file.
    frozen={name:p.read_evaluation(path) for name,path in roots.items()}
    baseline={x['image_id']:x for x in frozen['zero']}
    for records in frozen.values():
        for record in records:
            assert record['generated_tokens']==len(record['token_ids'])<=3084
            for field in ('prompt_token_ids','image_grid_thw','media_sha256','crop','seed','temperature','image_sha256'):
                assert record[field]==baseline[record['image_id']][field],field
    p.write(family_root/'all-evaluations-frozen.json',{name:dict(path=str(path),sha256=p.digest(path/'frozen.json')) for name,path in roots.items()})
    truth=p.load(TRUTH);partitions=p.load(ROOT/'cpu-03/evaluator-partitions.json')
    assert p.digest(TRUTH)==partitions['truth_sha256']
    assert len(truth)==18 and sum(len(x['objects']) for x in truth)==570
    assert len(set(map(tuple,partitions['hidden10'])))==57
    selected=[[x['image_id'],x['F']['annotation_id']] for x in p.load(ROOT/'cpu-04/credit-plan.json') if x['F']]
    assert len(selected)==16 and not set(map(tuple,selected)) & set(map(tuple,partitions['hidden10']))
    scored={name:assess_outputs(truth,partitions['hidden10'],records) for name,records in frozen.items()}
    outcomes=family_outcomes(scored,selected);contrasts={}
    comparisons=[(f'{later}-{step}','zero' if earlier=='zero' else f'{earlier}-{step}',f'{later}-{earlier}-{step}') for step in (4,8) for later,earlier in pairs]
    if 'S-16' in outcomes:comparisons=[('S-16','zero','S16-zero'),('S-16','S-8','S16-S8')]
    for later_name,earlier_name,contrast_name in comparisons:
        a=outcomes[earlier_name];b=outcomes[later_name]
        contrasts[contrast_name]={mode:{cohort:{part:{k:b['summary'][mode][cohort][part][k]-a['summary'][mode][cohort][part][k]
            for k in a['summary'][mode][cohort][part]} for part in ('metrics','burdens')} for cohort in ('combined','human13','refined5')} for mode in ('raw','category')}
    if 'S-16' in scored:
        p.write(family_root/'S16-versus-S8-ids.json',family_outcomes({'zero':scored['S-8'],'S-16':scored['S-16']},selected))
    p.write(family_root/'offline-results.json',dict(scored=scored,outcomes=outcomes,contrasts=contrasts,selected_FN_keys=selected,
        truth_sha256=p.digest(TRUTH),hidden_keys=partitions['hidden10'],denominator=570,
        limitations='Annotation-ID proxy; refined5 redraw uncertainty and prior exposure; overlap pairs are not physical negatives; no checkpoint selection.'))

def prepare_retained(output, reference_root, insertion_root):
    """CPU-only all-retained serialization; zero supplies input identities only."""
    visible=p.load(ROOT/'cpu-04/retained-10.json');zero=p.read_evaluation(ZERO);q=frontend()
    fields=('image_id','request_id','image_path','image_sha256','width','height','crop','prompt_token_ids','image_grid_thw','media_sha256')
    records={x['image_id']:{k:x[k] for k in fields} for x in zero}
    encodings=[];keys=[];total_tokens=total_visual=0
    for image in visible:
        i=image['image_id'];record=records[i];encoded,sequence,row_ids=retained_sequence(image,q)
        batch=p.native_request(record,p.load(p.POLICY),q.processor)
        assert list(batch.prompt_token_ids[0])==record['prompt_token_ids']==list(sequence.input_ids[:len(record['prompt_token_ids'])])
        assert list(batch.image_grids[0])==record['image_grid_thw']==list(encoded.image_encoding.image_grid_thw)
        assert batch.media_sha256[0]==record['media_sha256']
        positions=[a.causal_logits_position for a in sequence.atoms]
        ordered=sorted(image['objects'],key=lambda o:(o['bbox_2d'][0],o['bbox_2d'][1],o['coco_ann_id']))
        assert len(row_ids)==len(ordered) and len(positions)==len(set(positions))
        for oid,obj in zip(row_ids,ordered,strict=True):
            own=[a for a in sequence.atoms if a.object_id==oid]
            assert q.tokenizer.decode([a.token_id for a in own if a.token_type=='desc_text'],skip_special_tokens=False)==obj['desc']
            coords=[a.to_artifact_dict()['coordinate_target'] for a in own if a.token_type=='coordinate']
            assert len(coords)==4 and all(c['bbox']==obj['bbox_2d'] for c in coords)
            assert [c['slot_index'] for c in coords]==[0,1,2,3]
            keys.append([i,obj['coco_ann_id']])
        assert all(a.token_type!='eos' and sequence.input_ids[a.target_position]==a.token_id for a in sequence.atoms)
        visual=encoded.image_token_count;assert visual==record['image_grid_thw'][0]*record['image_grid_thw'][1]*record['image_grid_thw'][2]//4
        encodings.append(dict(image_id=i,prefix_source='rendered_retained_labels',row_ids=list(row_ids),
            annotation_ids=[o['coco_ann_id'] for o in ordered],input_ids=list(sequence.input_ids),
            input_sha256=p.stable(p.canonical(sequence.input_ids)),atoms=[a.to_artifact_dict() for a in sequence.atoms],
            positions=positions,tokens=len(sequence.input_ids),visual_tokens=visual,
            terminal_masked_positions=list(range(max(a.target_position for a in sequence.atoms)+1,len(sequence.input_ids))),
            grid=record['image_grid_thw'],media_sha256=record['media_sha256'],pixel_values_shape=list(batch.inputs['pixel_values'].shape),
            compact_logits_bf16_bytes=len(positions)*len(q.tokenizer)*2,compact_logits_fp32_bytes=len(positions)*len(q.tokenizer)*4))
        total_tokens+=len(sequence.input_ids);total_visual+=visual
    assert len(visible)==18 and len(keys)==len(set(map(tuple,keys)))==513
    assert total_visual*8==141280 and max(x['tokens'] for x in encodings)<=p.MAX_LENGTH
    p.write(output/'inputs.json',list(records.values()));p.write(output/'encodings.json',encodings)
    p.write(output/'annotation-keys.json',keys);p.write(output/'frontend.json',q.to_artifact_dict())
    refs=[ZERO,*[reference_root/f'evaluation-A-{step}' for step in (4,8)],*[insertion_root/f'evaluation-I-{step}' for step in (4,8)]]
    for root in refs:p.read_evaluation(root)
    bindings={str(root/'frozen.json'):p.digest(root/'frozen.json') for root in refs}
    for root in [reference_root/'train-A/checkpoint-0',insertion_root/'train-I/checkpoint-0']:
        bindings[str(root/'identity.json')]=p.digest(root/'identity.json')
        for name,sha in p.load(root/'identity.json').items():assert p.digest(root/name)==sha;bindings[str(root/name)]=sha
    for path in (p.POLICY,ROOT/'cpu-04/retained-10.json',reference_root/'family-plan.json',reference_root/'zero-equivalence.json',insertion_root/'family-plan.json'):
        bindings[str(path)]=p.digest(path)
    p.write(output/'control-reuse.json',dict(arm='S',reference_root=str(reference_root),insertion_root=str(insertion_root),sha256=bindings))
    prefix=['torchrun','--standalone','--nnodes=1','--nproc_per_node=8','-m','probes.rollout_row_credit']
    commands=[prefix+['family-train','--arm','S','--family-root',str(output),'--output',str(output/'train-S')]]
    commands += [prefix+['family-evaluate','--checkpoint',str(output/f'train-S/checkpoint-{step}'),'--family-root',str(output),'--output',str(output/f'evaluation-S-{step}')] for step in (4,8)]
    p.write(output/'family-plan.json',dict(status='CPU candidate; runtime not released',arm='S',prefix_source='rendered_retained_labels',
        retained_file=str(ROOT/'cpu-04/retained-10.json'),retained_sha256=p.digest(ROOT/'cpu-04/retained-10.json'),
        images=sorted(records),annotation_rows=513,seed=92711,updates=8,learning_rates=[1e-5,5e-6],image_mean=18,
        objective='rowmean(CE+.1type_gate+.01conditional_order_gate); object local closures included; EOS/terminal-only atoms masked',
        rank_jobs={str(rank):family_jobs(list(records),'S',rank) for rank in range(8)},training_forwards=144,
        training_input_tokens=total_tokens*8,training_visual_tokens=total_visual*8,natural_queries=36,natural_visual_tokens=35320,
        generated_token_bound=111024,max_input_tokens=max(x['tokens'] for x in encodings),max_selected_logits=max(len(x['positions']) for x in encodings),
        vocab_size=len(q.tokenizer),max_compact_logits_bf16_bytes=max(x['compact_logits_bf16_bytes'] for x in encodings),
        max_compact_logits_fp32_bytes=max(x['compact_logits_fp32_bytes'] for x in encodings),commands=commands,
        offline_command=['python','-m','probes.rollout_row_credit','family-offline','--family-root',str(output),'--reference-root',str(reference_root)]))


def prepare_insert(output, reference_root):
    """CPU frontend qualification of the fixed insertion targets and controls."""
    images,original,records=family_data()
    plans=insertion_plan(list(original.values()),list(records.values()))
    old={x['image_id']:x for x in p.load(ROOT/'cpu-04/encodings.json')}
    q=frontend();encodings=[];cuts=[];input_tokens=0;visual_tokens=0
    for plan in plans:
        i=plan['image_id'];image=images[i];record=records[i];before=original[i]
        assert plan['M']==before['M']
        batch=p.native_request(record,p.load(p.POLICY),q.processor)
        assert list(batch.prompt_token_ids[0])==record['prompt_token_ids']
        assert list(batch.image_grids[0])==old[i]['grid'] and batch.media_sha256[0]==old[i]['media_sha256']
        assert list(batch.inputs['pixel_values'].shape)==old[i]['pixel_values_shape']
        rows=[]
        for branch,values in (('M',plan['M']),('F',[plan['F']] if plan['F'] else [])):
            for row in values:
                seq=positive_sequence(image,record,row,q.tokenizer,branch=='F')
                rows.append(dict(branch=branch,input_sha256=p.stable(p.canonical(seq.input_ids)),tokens=len(seq.input_ids),
                                 atoms=[a.to_artifact_dict() for a in seq.atoms]))
        assert [x for x in rows if x['branch']=='M']==[x for x in old[i]['rows'] if x['branch']=='M']
        full=record['prompt_token_ids']+record['token_ids'];fn_tokens=len(full)
        if plan['F']:
            for key in ('annotation_id','description','bbox','token_ids'):assert plan['F'][key]==before['F'][key]
            cut=plan['insertion']['cut'];seq=positive_sequence(image,record,plan['F'],q.tokenizer,True)
            assert list(seq.input_ids)==record['prompt_token_ids']+record['token_ids'][:cut]+before['F']['token_ids']
            assert [a.token_id for a in seq.atoms]==before['F']['token_ids']
            fn_tokens=len(seq.input_ids)
            cuts.append(dict(image_id=i,annotation_id=plan['F']['annotation_id'],target_bbox=plan['F']['bbox'],
                             target_tokens=plan['F']['token_ids'],**plan['insertion'],input_tokens=fn_tokens,
                             terminal_input_tokens=next(x['tokens'] for x in old[i]['rows'] if x['branch']=='F')))
        encodings.append(dict(old[i],rows=rows,selected_logit_positions={'I':old[i]['selected_logit_positions']['B']},
                              original_input_tokens=len(full),F_input_tokens=fn_tokens))
        input_tokens+=len(full)+fn_tokens;visual_tokens+=2*old[i]['visual_tokens']
    assert len(cuts)==16 and sum(x['successor_M_credit'] for x in cuts)==9
    assert visual_tokens*8==282560
    p.write(output/'credit-plan.json',plans);p.write(output/'encodings.json',encodings);p.write(output/'cuts.json',cuts)
    # Explicit immutable reference binding; old readback validates its frozen shards.
    references=[ZERO,*[reference_root/f'evaluation-{arm}-{step}' for arm in 'AB' for step in (4,8)]]
    for root in references:p.read_evaluation(root)
    bindings={str(root/'frozen.json'):p.digest(root/'frozen.json') for root in references}
    for root in [reference_root/'train-A/checkpoint-0',reference_root/'train-B/checkpoint-0']:
        bindings[str(root/'identity.json')]=p.digest(root/'identity.json')
        for name,sha in p.load(root/'identity.json').items():
            assert p.digest(root/name)==sha;bindings[str(root/name)]=sha
    for path in (p.POLICY,reference_root/'family-plan.json',reference_root/'zero-equivalence.json',
                 ROOT/'cpu-04/retained-10.json',ROOT/'cpu-04/credit-plan.json',ROOT/'cpu-04/encodings.json'):
        bindings[str(path)]=p.digest(path)
    p.write(output/'control-reuse.json',dict(reference_root=str(reference_root),sha256=bindings,
        semantics='Reuse accepted A/B and zero; same M atoms/history,18-image mean,seed,teacher,optimizer,LRs and losses. Only F prefix changes.'))
    prefix=['torchrun','--standalone','--nnodes=1','--nproc_per_node=8','-m','probes.rollout_row_credit']
    commands=[prefix+['family-train','--arm','I','--family-root',str(output),'--output',str(output/'train-I')]]
    commands += [prefix+['family-evaluate','--checkpoint',str(output/f'train-I/checkpoint-{step}'),
                        '--family-root',str(output),'--output',str(output/f'evaluation-I-{step}')] for step in (4,8)]
    p.write(output/'family-plan.json',dict(status='CPU candidate; runtime not released',arm='I',reference_root=str(reference_root),
        images=sorted(images),seed=92711,updates=8,learning_rates=[1e-5,5e-6],F_coefficient=1,original_image_mean=18,
        rank_jobs={str(rank):family_jobs(list(images),'I',rank) for rank in range(8)},
        training_forwards=288,training_input_tokens=input_tokens*8,training_visual_tokens=visual_tokens*8,
        natural_queries=36,natural_visual_tokens=35320,generated_token_bound=111024,commands=commands,
        offline_command=['python','-m','probes.rollout_row_credit','family-offline','--family-root',str(output),'--reference-root',str(reference_root)]))


def evaluate_partition(truth, hidden_keys, records):
    """Offline only, membership is composite key, never sign of annotation ID."""
    hidden={tuple(x) for x in hidden_keys}; predictions,_=p.candidates(records); result=[]
    for image in truth:
        refs=[dict(owner_id=str(o['coco_ann_id']),reference_coord_bins_1000=o['bbox_2d']) for o in image['objects']]
        pool=[r for r in predictions if r['image_id']==image['image_id']]; by={r['prediction_id']:r for r in pool}
        matches=one_to_one_matches(refs,pool,.5); covered={'hidden':[],'retained':[]}; disagree=[]
        for m in matches:
            o=image['objects'][m['reference_index']]; key=(image['image_id'],o['coco_ann_id'])
            if o['desc']==by[m['prediction_id']]['description']: covered['hidden' if key in hidden else 'retained'].append(list(key))
            else: disagree.append(m)
        result.append(dict(image_id=image['image_id'],cohort=image['cohort'],covered=covered,category_disagreements=disagree,
                           hidden_denominator=sum((image['image_id'],o['coco_ann_id']) in hidden for o in image['objects']),
                           retained_denominator=sum((image['image_id'],o['coco_ann_id']) not in hidden for o in image['objects'])))
    return result


def prepare(output):
    release=ROOT/'lead-release-01.json'
    assert p.digest(release)==RELEASE_SHA
    for path,sha in p.load(release)['sha256'].items(): assert p.digest(path)==sha,path
    truth=p.load(TRUTH); visible,hidden=split_reference(truth,10); v20,h20=split_reference(truth,20)
    assert len(hidden)==57 and len(h20)==114 and set(map(tuple,hidden))<=set(map(tuple,h20))
    p.write(output/'retained-10.json',visible);p.write(output/'retained-20.json',v20)
    p.write(output/'evaluator-partitions.json',dict(hidden10=hidden,hidden20=h20,truth_sha256=p.digest(TRUTH)))
    records=p.read_evaluation(ZERO);q=frontend();plans=credit_plan(visible,records,q.tokenizer)
    encodings=[]
    for image,plan in zip(visible,plans,strict=True):
        r=next(r for r in records if r['image_id']==image['image_id'])
        batch=p.native_request(r,p.load(p.POLICY),q.processor)
        assert list(batch.prompt_token_ids[0])==r['prompt_token_ids']
        assert list(batch.image_grids[0])==r['image_grid_thw'] and batch.media_sha256[0]==r['media_sha256']
        rows=[]
        for branch,values in (('M',plan['M']),('F',[plan['F']] if plan['F'] else [])):
            for row in values:
                seq=positive_sequence(image,r,row,q.tokenizer,branch=='F')
                rows.append(dict(branch=branch,input_sha256=p.stable(p.canonical(seq.input_ids)),tokens=len(seq.input_ids),atoms=[a.to_artifact_dict() for a in seq.atoms]))
        encodings.append(dict(image_id=image['image_id'],prompt_tokens=len(r['prompt_token_ids']),generated_tokens=len(r['token_ids']),
                              visual_tokens=r['image_grid_thw'][0]*r['image_grid_thw'][1]*r['image_grid_thw'][2]//4,
                              grid=r['image_grid_thw'],media_sha256=r['media_sha256'],rows=rows,pixel_values_shape=list(batch.inputs['pixel_values'].shape),
                              original_pixels=image['width']*image['height'],
                              selected_logit_positions={a:objective_positions(plan,r,a) for a in 'ABC'}))
    p.write(output/'credit-plan.json',plans);p.write(output/'encodings.json',encodings)
    # Freeze consumer artifacts before separate evaluator readback.
    p.write(output/'consumer-freeze.json',{str(output/n):p.digest(output/n) for n in ('retained-10.json','credit-plan.json','encodings.json')})
    p.write(output/'zero-partition-evaluation.json',evaluate_partition(truth,hidden,records))
    counts=[dict(image_id=x['image_id'],cohort=x['cohort'],retained=x['retained_count'],M=len(x['M']),F=int(x['F'] is not None),
                 known_FN=len(x['known_fn_ids']),D=len(x['D']),G=len(x['G']),invalid_occurrences=len(x['negative_evidence']['complete_geometry_invalid']),
                 malformed=len(x['negative_evidence']['malformed_or_censored']),fn_ineligible=x['fn_ineligible_reason']) for x in plans]
    p.write(output/'counts.json',counts)
    return counts


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','prepare-insert','prepare-retained','slice','reload','family-train','family-evaluate','family-offline','checkpoint-gate']);parser.add_argument('--output',type=Path,default=ROOT/'cpu-01')
    parser.add_argument('--cpu-root',type=Path,default=ROOT/'cpu-03');parser.add_argument('--arm',choices=list('ABCIS'),default='C')
    parser.add_argument('--training-root',type=Path)
    parser.add_argument('--family-root',type=Path,default=FAMILY);parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--reference-root',type=Path)
    parser.add_argument('--insertion-root',type=Path)
    parser.add_argument('--updates',type=int,default=8)
    args=parser.parse_args()
    if args.command=='prepare':
        args.output.mkdir(parents=True,exist_ok=True);print(p.canonical(prepare(args.output)))
    elif args.command=='prepare-retained':
        if args.reference_root is None or args.insertion_root is None:parser.error('prepare-retained requires reference-root and insertion-root')
        args.output.mkdir(parents=True,exist_ok=True);prepare_retained(args.output,args.reference_root,args.insertion_root)
    elif args.command=='prepare-insert':
        if args.reference_root is None:parser.error('prepare-insert requires --reference-root')
        args.output.mkdir(parents=True,exist_ok=True);prepare_insert(args.output,args.reference_root)
    elif args.command=='family-train':family_train(args.output,args.arm,args.family_root,args.updates)
    elif args.command=='checkpoint-gate':
        reuse=p.load(args.family_root/'control-reuse.json')
        bindings={**reuse['sha256'],**p.load(args.family_root/'qualification.json')['sha256']}
        p.write(args.output,checkpoint_gate(args.checkpoint,Path(reuse['retained_root'])/'train-S/checkpoint-8',bindings))
    elif args.command=='family-evaluate':
        if args.checkpoint is None:parser.error('family-evaluate requires --checkpoint')
        family_evaluate(args.checkpoint,args.output,args.family_root)
    elif args.command=='family-offline':family_offline(args.family_root,args.reference_root)
    else:
        if args.command=='reload' and args.training_root is None: parser.error('reload requires --training-root')
        slice_run(args.output,args.cpu_root,args.arm,args.training_root if args.command=='reload' else None)


if __name__=='__main__': main()
