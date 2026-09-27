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
        history=history[:-1]+row['token_ids']
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



def forward_credit(q, model, plan, record, image, vocab, arm, fn=False):
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
        detail={}
    else:
        loss,detail=image_objective(arm,dict(plan,F=None),record,image,q.tokenizer,vocab,logits,positions)
    evidence=dict(image_id=image['image_id'],branch='F' if fn else 'M_D_G',tokens=len(full),
                  input_sha256=p.stable(p.canonical(list(full))),positions=list(positions),
                  logits_sha256=p.tensor_hash(logits),loss=float(loss.detach()),
                  row_probabilities=detail.get('row_probabilities',[]))
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
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','slice','reload']);parser.add_argument('--output',type=Path,default=ROOT/'cpu-01')
    parser.add_argument('--cpu-root',type=Path,default=ROOT/'cpu-03');parser.add_argument('--arm',choices=list('ABC'),default='C')
    parser.add_argument('--training-root',type=Path)
    args=parser.parse_args()
    if args.command=='prepare':
        args.output.mkdir(parents=True,exist_ok=True);print(p.canonical(prepare(args.output)))
    else:
        if args.command=='reload' and args.training_root is None: parser.error('reload requires --training-root')
        slice_run(args.output,args.cpu_root,args.arm,args.training_root if args.command=='reload' else None)


if __name__=='__main__': main()
