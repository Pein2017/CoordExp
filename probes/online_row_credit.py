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


def credit(image, record, tokenizer, producer, redirect_enabled=True):
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
    for duplicate in ((o for o in rows if not o['first']) if redirect_enabled else ()):
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


def bridge_credit(image, record, tokenizer, producer, policy, schema_geometry=False):
    assert policy in ('local','chain')
    """Freeze original matching/vetoes, then insert every admissible retained miss."""
    plan=credit(image,record,tokenizer,producer,redirect_enabled=False)
    if schema_geometry:plan['schema_geometry']=schema_geometry_errors(record,tokenizer,plan['observations'])
    covered={x['annotation_id'] for x in plan['M']};dispositions=[];insertions=[]
    valid=[x for x in plan['observations'] if x['valid']]
    eos=tokenizer.convert_tokens_to_ids('<|im_end|>');tokens=record['token_ids']
    ends=[j for j,t in enumerate(tokens) if t==eos]
    # Same certified terminal predicate as maintained rollout-row-credit selection;
    # complete invalid rows are observations, not malformed/censored spans.
    terminal=(record['stop_reason'] in ('im_end','eos') and ends==[len(tokens)-1]
        and (tokens==[eos] or not plan['malformed']) and (len(tokens)==1 or tokens[-2]==tokenizer.convert_tokens_to_ids('<|box_end|>')))
    for obj in sorted(image['objects'],key=lambda x:(*x['bbox_2d'][:2],int(x['coco_ann_id']))):
        ann=obj['coco_ann_id'];reason='matched' if ann in covered else None
        if reason is None and any(x['description']==obj['desc'] and iou_xyxy(x['bbox'],obj['bbox_2d'])>=.5 for x in valid):reason='same_category_supported'
        successor=next((x for x in plan['M'] if tuple(x['bbox'][:2])>tuple(obj['bbox_2d'][:2])),None)
        if reason is None:
            if successor is None and not terminal:reason='unsupported_boundary'
            else:
                cut=successor['positions'][0] if successor is not None else len(tokens)-1
                ids=tokenizer.encode(r.render_row(image,obj).assistant_content_text,add_special_tokens=False)
                insertions.append(dict(annotation_id=ann,description=obj['desc'],bbox=obj['bbox_2d'],token_ids=ids,cut=cut,
                    boundary='successor' if successor is not None else 'terminal',successor_order=None if successor is None else successor['order']))
                reason='inserted'
        dispositions.append(dict(annotation_id=ann,reason=reason))
    eligible=insertions
    m=len(plan['M']);k=len(insertions);repair=None
    if k:
        assert len({x['annotation_id'] for x in insertions})==k
        repaired=[];mapping=[];added=[]
        for j,token in enumerate(tokens):
            for row in (x for x in insertions if x['cut']==j):
                start=len(repaired);repaired.extend(row['token_ids'])
                added.append(dict(row,positions=list(range(start,len(repaired)))))
            mapping.append(len(repaired));repaired.append(token)
        assert [repaired[j] for j in mapping]==tokens
        shifted=[dict(row,positions=[mapping[j] for j in row['positions']],
            coordinate_positions=[mapping[j] for j in row['coordinate_positions']]) for row in plan['M']]
        repair=dict(B=added,M=shifted,original_to_repaired=mapping,token_ids=repaired)
    return dict(plan,bridge=repair,bridge_dispositions=dispositions,m=m,k=k,n=m+k,eligible=eligible,policy=policy,terminal_certified=terminal)


def bridge_rows(record, plan, branch_index=None):
    """Rows and exact token history for one real repaired forward."""
    assert plan['k']>0
    if plan['policy']=='chain':
        assert branch_index is None
        repair=plan['bridge']
        return repair['token_ids'],repair['B']+repair['M'],[1/plan['n']]*plan['n'],plan['k']
    assert plan['policy']=='local' and isinstance(branch_index,int) and 0<=branch_index<plan['k']
    b=plan['eligible'][branch_index];cut=b['cut'];tokens=record['token_ids'][:cut]+b['token_ids']
    rows=[dict(b,positions=list(range(cut,len(tokens))))];weights=[1/plan['n']]
    if b['successor_order'] is not None:
        c=next(x for x in plan['M'] if x['order']==b['successor_order'])
        assert c['positions'][0]==cut
        start=len(tokens);tokens.extend(record['token_ids'][j] for j in c['positions'])
        shift=start-cut
        rows.append(dict(c,positions=[j+shift for j in c['positions']],coordinate_positions=[j+shift for j in c['coordinate_positions']]))
        q=sum(x['successor_order']==c['order'] for x in plan['eligible'])
        weights.append(1/(plan['n']*q))
    return tokens,rows,weights,1


def bridge_sequences(image, record, plan, tokenizer, branch_index=None):
    assert bridge_credit(image,record,tokenizer,plan['producer'],plan['policy'],'schema_geometry' in plan)==plan, 'bridge plan drift'
    tokens,rows,_,_=bridge_rows(record,plan,branch_index)
    sequences=[r.positive_sequence(image,dict(record,token_ids=tokens),row,tokenizer) for row in rows]
    assert all(list(seq.input_ids)==record['prompt_token_ids']+tokens for seq in sequences)
    return sequences


def bridge_trace_plan(plan, arm):
    assert arm in ('local','chain') and plan['policy']==arm
    successors={x['successor_order'] for x in plan['eligible']}
    return dict(plan,M=[x for x in plan['M'] if not plan['k'] or (arm=='local' and x['order'] not in successors)])


def bridge_objective(logits, positions, sequences, plan, arm, vocab, branch_index=None):
    import torch
    assert arm in ('local','chain') and plan['policy']==arm and plan['n']>0 and plan['k']>0
    if arm=='chain':
        assert branch_index is None
        weights=[1/plan['n']]*plan['n'];nb=plan['k']
    else:
        assert isinstance(branch_index,int) and 0<=branch_index<plan['k']
        b=plan['eligible'][branch_index];weights=[1/plan['n']];nb=1
        if b['successor_order'] is not None:
            weights.append(1/(plan['n']*sum(x['successor_order']==b['successor_order'] for x in plan['eligible'])))
    assert len(sequences)==len(weights)
    expected=tuple(sorted({a.causal_logits_position for seq in sequences for a in seq.atoms}))
    assert tuple(positions)==expected
    values=[p.image_loss(logits,seq,vocab,positions)[0]*weight for seq,weight in zip(sequences,weights)]
    terms={'B':torch.stack(values[:nb]).sum(),'M_relocated':torch.stack(values[nb:]).sum() if len(values)>nb else logits.sum()*0}
    return sum(terms.values()),terms


def bridge_metadata(plan, arm, branch, branch_index=None):
    return dict(policy=arm,plan_sha256=identity(plan),branch_index=branch_index,m=plan['m'],k=plan['k'],n=plan['n'],eligible=len(plan['eligible']),
        successor_multiplicity={str(c):sum(x['successor_order']==c for x in plan['eligible']) for c in sorted({x['successor_order'] for x in plan['eligible'] if x['successor_order'] is not None})},
        matched_share=plan['m']/plan['n'] if plan['n'] else 0,insertion_share=plan['k']/plan['n'] if plan['n'] else 0,
        selected_M_orders=[x['order'] for x in bridge_trace_plan(plan,arm)['M']] if branch=='trace' else [])


def bridge_row_evidence(record,plan,sequences,branch_index):
    _,rows,weights,nb=bridge_rows(record,plan,branch_index)
    return [dict(kind='B' if j<nb else 'M',annotation_id=row['annotation_id'],weight=weight,atoms=[a.to_artifact_dict() for a in seq.atoms])
        for j,(row,seq,weight) in enumerate(zip(rows,sequences,weights))]


def verify_bridge_forward(evidence, image, record, plan, tokenizer, arm):
    assert bridge_credit(image,record,tokenizer,record['producer'],arm,'schema_geometry' in plan)==plan
    branch=evidence['branch'];assert branch in ('trace','bridge');index=evidence['bridge']['branch_index']
    selected=bridge_trace_plan(plan,arm) if branch=='trace' else plan
    if branch=='trace':
        assert index is None
        full=record['prompt_token_ids']+record['token_ids'];positions=trace_positions(selected,record)
        rows=[dict(order=row['order'],atoms=[a.to_artifact_dict() for a in r.positive_sequence(image,record,row,tokenizer).atoms]) for row in selected['M']]
    else:
        sequences=bridge_sequences(image,record,plan,tokenizer,index);full=list(sequences[0].input_ids)
        positions=tuple(sorted({a.causal_logits_position for seq in sequences for a in seq.atoms}))
        rows=bridge_row_evidence(record,plan,sequences,index)
    if branch=='trace':
        assert evidence.get('schema_geometry')==plan.get('schema_geometry')
    assert evidence['row_losses']==rows
    assert evidence['input_sha256']==identity(list(full)) and evidence['positions']==list(positions)
    assert evidence['tokens']==len(full) and evidence['producer']==record['producer'] and evidence['raw_identity']==record['raw_identity']
    assert evidence['bridge']==bridge_metadata(plan,arm,branch,index)
    expected={'M','legal','Gmax_unweighted','Gmax_weighted'} if branch=='trace' else {'B','M_relocated'}
    assert set(evidence['terms'])==expected


def verify_bridge_schedule(forwards, image_ids, rank, plans, arm):
    expected=jobs(image_ids,rank,plans,insertion_policy=arm)
    assert [(x['image_id'],x['branch'],x['bridge']['branch_index'],x['sync'],x['image_weight']) for x in forwards]==[(x['image_id'],x['branch'],x['branch_index'],x['sync'],x['weight']) for x in expected]


def require_bridge_supervision(plans, output, version):
    """Global plans are made from the already-saved same-version all-rank records."""
    if not any(plan['n'] or any(legal_slots(row) for row in plan['observations']) or plan.get('schema_geometry',{}).get('union') for plan in plans.values()):
        p.write(output/f'no-supervision-{version}.json',dict(status='no_supervision',version=version,
            reason='No semantic rows or certified original coordinate contexts; optimizer not invoked',
            plans={str(i):identity(plan) for i,plan in plans.items()}))
        raise RuntimeError('no-supervision: stopped before optimizer action')


def redirect_sequence(image, record, target, tokenizer):
    cut = target['prefix_cut']
    history = record['token_ids'][:cut] + target['token_ids']
    assert history[:cut] == record['token_ids'][:cut]
    assert semantic_site(target['token_ids'],target['negative_ids'],tokenizer) == target['site']
    return r.positive_sequence(image,dict(record,token_ids=history),target,tokenizer)


def legal_slots(row):
    return [(row['coordinate_positions'][j], 0 if j<2 else row['bbox'][j-2]+1, 999 if j<2 else 1000)
            for j in range(4) if j<2 or row['bbox'][j-2]<999]


def schema_geometry_binding(root, enabled, arm):
    assert isinstance(enabled,bool)
    assert p.load(root/'qualification.json').get('schema_geometry',False)==enabled, 'schema geometry recipe drift'
    if enabled:assert arm in ('local','chain')


def schema_geometry_errors(record, tokenizer, rows):
    """Strict literal wrapper eligibility only; never manufacture a parsed box."""
    ids=record['token_ids'];coord={tokenizer.convert_tokens_to_ids(f'<|coord_{i}|>'):i for i in range(1000)}
    start,end,box,close=[tokenizer.convert_tokens_to_ids(x) for x in
        ('<|object_ref_start|>','<|object_ref_end|>','<|box_start|>','<|box_end|>')]
    structural=(set(tokenizer.all_special_ids)-set(coord))|{start,end,box,close}
    structural.update(v for k,v in tokenizer.get_added_vocab().items() if k.startswith('<|') and v not in coord)
    old=list(erroneous_slots(rows));new=[];dispositions=[];active=None;ambiguous=False
    for j,t in enumerate(ids):
        if t==start:
            if active is not None:ambiguous=True
            else:active=j;ambiguous=False
        if t!=close or active is None:continue
        begin=active;active=None;body=ids[begin+1:j];reason=None
        if ambiguous:reason='nested_wrapper'
        elif body.count(end)!=1:reason='object_ref_delimiter'
        else:
            split=body.index(end);description=body[:split];tail=body[split+1:]
            if not description or any(x in structural or x in coord for x in description):reason='ambiguous_description'
            elif len(tail)!=5 or tail[0]!=box:reason='not_four_slots'
            elif any(x in structural for x in tail[1:]):reason='structural_slot'
            elif all(x in coord for x in tail[1:]):reason='already_coordinate_only'
            else:
                slots=tail[1:];first=begin+1+split+2
                for slot,token in enumerate(slots):
                    pos=first+slot;lo=0;hi=999
                    if slot>=2:
                        own=coord.get(slots[slot-2])
                        if own is None or own==999:
                            dispositions.append(dict(position=pos,reason='unknown_start' if own is None else 'empty_end',wrapper=begin));continue
                        lo=own+1;hi=1000
                    value=coord.get(token)
                    if value is None or not lo<=value<hi:
                        new.append((pos,lo,hi));dispositions.append(dict(position=pos,wrapper=begin,reason='type' if value is None else 'geometry',bounds=[lo,hi],token_id=token))
                reason='eligible_four_slots'
        dispositions.append(dict(wrapper=begin,end=j,reason=reason))
    if active is not None:dispositions.append(dict(wrapper=active,reason='incomplete_or_censored'))
    union={pos:(pos,lo,hi) for pos,lo,hi in old}
    for item in new:
        assert item[0] not in union or union[item[0]]==item
        union[item[0]]=item
    return dict(old=[list(x) for x in old],new=[list(x) for x in new],union=[list(union[k]) for k in sorted(union)],dispositions=dispositions,
                source_raw_identity=record['raw_identity'],producer=record['producer'])


def trace_positions(plan, record):
    n = len(record['prompt_token_ids'])
    targets = {n+j for row in plan['M'] for j in row['positions']}
    targets.update(n+j for row in plan['observations'] for j,_,_ in legal_slots(row))
    targets.update(n+j for j,_,_ in plan.get('schema_geometry',{}).get('union',[]))
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


def greedy_geometry_objective(logits, positions, rows, prompt_length, coordinate_ids, errors=None):
    import torch
    assert len(set(positions))==len(positions)
    values=[max_geometry_margin(logits[0,positions.index(prompt_length+pos-1)],coordinate_ids[lo:hi])
            for pos,lo,hi in (erroneous_slots(rows) if errors is None else errors)]
    return torch.stack(values).mean() if values else logits.sum()*0


def trace_objective(logits, positions, plan, record, image, tokenizer, vocab, geometry_weight=0, matched_denominator=None):
    import torch
    if 'schema_geometry' in plan:
        assert plan['schema_geometry']==schema_geometry_errors(record,tokenizer,plan['observations']), 'schema causal bounds drift'
    assert tuple(positions)==trace_positions(plan,record), 'wrong causal positions'
    assert plan['raw_identity']==record['raw_identity'] and plan['producer']==record['producer']
    values = [p.image_loss(logits,r.positive_sequence(image,record,row,tokenizer),vocab,positions)[0] for row in plan['M']]
    m = (torch.stack(values).sum()/matched_denominator if matched_denominator is not None else torch.stack(values).mean()) if values else logits.sum()*0
    legal = legal_objective(logits,positions,plan['observations'],len(record['prompt_token_ids']),vocab.coordinate)
    assert geometry_weight in (0,.1)
    if geometry_weight:
        g=greedy_geometry_objective(logits,positions,plan['observations'],len(record['prompt_token_ids']),vocab.coordinate,plan.get('schema_geometry',{}).get('union'))
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


def witness_bank(source, image_ids, tokenizer):
    """Prediction-only earliest version/position; no annotation consumer."""
    selected = {}
    for version in range(17):
        for record in frozen_records(source/f'rollout-{version}', image_ids):
            i = record['image_id']
            if i in selected: continue
            rows, _ = observations(record, tokenizer)
            errors = erroneous_slots(rows)
            if not errors: continue
            position, lo, hi = min(errors)
            prefix = record['prompt_token_ids']+record['token_ids'][:position]
            selected[i] = dict(image_id=i, source_version=version, record=record,
                generated_position=position, legal_range=[lo,hi], input_ids=prefix,
                input_sha256=identity(prefix), causal_position=len(prefix)-1)
    return dict(kind='fixed_geometry_witness', source=str(source), images=[selected[i] for i in sorted(selected)])


def witness_inputs(entry, tokenizer):
    record=entry['record'];verify_producer([record],record['producer'],[entry['image_id']])
    assert record['producer']['update']==entry['source_version']
    rows,_=observations(record,tokenizer)
    position,lo,hi=min(erroneous_slots(rows))
    full=record['prompt_token_ids']+record['token_ids'][:position]
    assert entry['generated_position']==position and entry['legal_range']==[lo,hi]
    assert entry['input_ids']==full and entry['input_sha256']==identity(full)
    assert entry['causal_position']==len(full)-1
    return full,[len(full)-1]


def witness_binding(root, weight, bank_sha256):
    assert weight in (0,.1)
    spec=p.load(root/'qualification.json').get('witness')
    if spec is None:
        assert weight==0 and bank_sha256 is None
        return None,None
    assert bank_sha256==spec['sha256']==p.digest(Path(spec['path'])), 'wrong witness bank identity'
    bank=p.load(Path(spec['path']));assert bank['kind']=='fixed_geometry_witness'
    entries={e['image_id']:e for e in bank['images']}
    assert len(entries)==len(bank['images'])==4
    assert sorted(entries)==spec['image_ids'] and bank['source']==spec['source']
    return dict(weight=weight,bank_sha256=bank_sha256,bank_path=spec['path']),entries


def witness_forward(q, model, batch, entry, vocab, producer, weight, diagnostic=False):
    import torch,time
    from contextlib import nullcontext
    from src.qwen.native import exact_history_inputs
    assert weight in (0,.1)
    begin=time.monotonic()
    full,positions=witness_inputs(entry,q.tokenizer)
    kwargs=exact_history_inputs(q.model,batch.inputs,[full],pad_token_id=q.tokenizer.pad_token_id)
    kwargs['logits_to_keep']=torch.tensor(positions,device='cuda')
    with torch.no_grad() if diagnostic else nullcontext():
        with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(**kwargs).logits
        assert logits.shape[1]==1
        z=logits[0,0].float();lo,hi=entry['legal_range'];legal=list(vocab.coordinate[lo:hi])
        margin=max_geometry_margin(z,legal);loss=weight*margin
        assert torch.isfinite(z).all() and torch.isfinite(margin)
    detail=dict(image_id=entry['image_id'],branch='witness',producer=producer,
        source_producer=entry['record']['producer'],source_raw_identity=entry['record']['raw_identity'],
        entry_sha256=identity(entry),input_sha256=identity(full),positions=positions,
        tokens=len(full),visual_tokens=__import__('math').prod(entry['record']['image_grid_thw'])//4,
        weight=weight,loss=float(loss.detach()),margin=float(margin.detach()),logits_sha256=p.tensor_hash(logits))
    if diagnostic:
        z=z.detach();mask=torch.ones_like(z,dtype=torch.bool);mask[legal]=False
        ml=float(z[legal].amax());mi=float(z[mask].amax())
        detail.update(max_legal=ml,max_illegal=mi,legal_minus_illegal=ml-mi,
            argmax_token=int(z.argmax()),argmax_legal=int(z.argmax()) in legal,
            argmax_ties=int((z==z.max()).sum()),diagnostic=True)
    else:detail['logit_derivatives']=diagnostics({'witness_unweighted':margin,'witness_weighted':loss},logits)
    detail['seconds']=time.monotonic()-begin
    return loss,detail


def witness_diagnostics(q, batches, entries, vocab, producer):
    """Same prefix path, no model backward; preserve pre-existing gradients."""
    before=parameter_identity(q.model)
    assert identity(before)==producer['parameter_sha256'], 'stale witness diagnostic producer'
    gradients={n:None if x.grad is None else p.tensor_hash(x.grad) for n,x in q.model.named_parameters()}
    rows=[witness_forward(q,q.model,batches[i],entries[i],vocab,producer,0,True)[1] for i in sorted(entries) if i in batches]
    assert parameter_identity(q.model)==before
    assert gradients=={n:None if x.grad is None else p.tensor_hash(x.grad) for n,x in q.model.named_parameters()}
    return dict(producer=producer,parameters_sha256=identity(before),unchanged_parameters_and_gradients=True,rows=rows)


def jobs(image_ids, rank, plans, preservation_weight=0, witness_ids=(), insertion_policy=None):
    assert len(image_ids)==len(set(image_ids))==18 and 0<=rank<8
    assert preservation_weight in (0,.25)
    result = []
    if insertion_policy is not None:
        assert insertion_policy in ('local','chain') and preservation_weight==0 and not witness_ids
        for i in sorted(image_ids)[rank::8]:
            result.append(dict(image_id=i,branch='trace',branch_index=None,weight=8/18))
            if plans[i]['k']:
                result += [dict(image_id=i,branch='bridge',branch_index=j,weight=8/18) for j in (range(plans[i]['k']) if insertion_policy=='local' else [None])]
        return [dict(x,sync=j==len(result)-1) for j,x in enumerate(result)]
    for i in sorted(image_ids)[rank::8]:
        result += [dict(image_id=i,branch=b,weight=8/18) for b in
                   (['trace']+(['redirect'] if plans[i]['redirect'] else [])+(['P0'] if preservation_weight else [])+
                    (['witness'] if i in witness_ids else [])+['R'])]
    return [dict(x,sync=j==len(result)-1) for j,x in enumerate(result)]


def physical_jobs(logical, microbatch=1):
    assert microbatch in (1,4)
    if microbatch==1:return logical
    result=[]
    for job in logical:
        if job['branch']=='bridge' and job.get('branch_index') is not None:
            if result and result[-1]['branch']=='bridge_group' and result[-1]['image_id']==job['image_id'] and len(result[-1]['members'])<microbatch:
                assert result[-1]['members'][-1]+1==job['branch_index']
                result[-1]['members'].append(job['branch_index'])
            else:result.append(dict(image_id=job['image_id'],branch='bridge_group',members=[job['branch_index']],weight=job['weight']))
        else:result.append(dict(job))
    return [dict(j,sync=i==len(result)-1) for i,j in enumerate(result)]


def execution_binding(root, arm, microbatch, checkpointing):
    assert microbatch in (1,4) and isinstance(checkpointing,bool)
    assert microbatch==1 or arm=='local'
    setting=dict(arm=arm,microbatch=microbatch,activation_checkpointing=checkpointing)
    spec=p.load(root/'qualification.json').get('execution')
    if spec is None:
        assert microbatch==1 and checkpointing
        return None
    assert setting in spec['profiles'], 'execution profile drift'
    return setting


def set_checkpointing(model, enabled, enable_inputs=True):
    if enabled:model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    else:model.gradient_checkpointing_disable()
    if enable_inputs:model.enable_input_require_grads()
    assert bool(model.is_gradient_checkpointing)==enabled


def logical_forwards(physical):
    return [row for group in physical for row in (group['logical'] if group['branch']=='bridge_group' else [group])]


def verify_physical_forwards(physical, image_ids, rank, plans, records, images, tokenizer, arm, microbatch):
    schedule=physical_jobs(jobs(image_ids,rank,plans,insertion_policy=arm),microbatch)
    assert len(physical)==len(schedule)
    for f,job in zip(physical,schedule):
        assert (f['image_id'],f['branch'],f['sync'],f['image_weight'])==(job['image_id'],job['branch'],job['sync'],job['weight'])
        i=f['image_id']
        if f['branch']=='bridge_group':
            assert f['members']==job['members'] and [x['bridge']['branch_index'] for x in f['logical']]==job['members']
            assert all(x['image_id']==i for x in f['logical'])
            lengths=[x['tokens'] for x in f['logical']];counts=[len(x['positions']) for x in f['logical']]
            from src.qwen.native import padded_histories
            full=[records[i]['prompt_token_ids']+bridge_rows(records[i],plans[i],j)[0] for j in job['members']]
            ids,mask=padded_histories(full,pad_token_id=tokenizer.pad_token_id)
            assert f['native_sha256']['input_ids']==p.tensor_hash(ids) and f['native_sha256']['attention_mask']==p.tensor_hash(mask)
            assert f['logits_shape']==[len(lengths),max(counts)+1,len(tokenizer)]
            assert f['shape']==dict(batch=len(lengths),lengths=lengths,left_padding=[max(lengths)-n for n in lengths],padded_tokens=len(lengths)*max(lengths),unpadded_tokens=sum(lengths),compact_rows=len(lengths)*(max(counts)+1),selected_rows=sum(counts))
            assert abs(f['loss']-sum(x['loss'] for x in f['logical']))<=1e-5*max(1,abs(f['loss']))
        else:
            assert f['bridge']['branch_index']==job['branch_index']
        for row in f['logical'] if f['branch']=='bridge_group' else [f]:
            verify_bridge_forward(row,images[i],records[i],plans[i],tokenizer,arm)


def group_forward(q, model, batch, image, record, plan, vocab, members, diagnostic=True):
    import torch
    from src.qwen.native import combine_singleton_native_inputs,exact_history_inputs,select_compact_replay_logits
    assert plan['policy']=='local' and 1<=len(members)<=4
    assert members==list(range(members[0],members[0]+len(members)))
    sequences=[bridge_sequences(image,record,plan,q.tokenizer,j) for j in members]
    full=[list(seqs[0].input_ids) for seqs in sequences]
    positions=[tuple(sorted({a.causal_logits_position for seq in seqs for a in seq.atoms})) for seqs in sequences]
    counts=[len(pos) for pos in positions]
    for ids,pos,n in zip(full,positions,counts):
        assert len(ids)<=p.MAX_LENGTH and pos==tuple(range(len(ids)-n-1,len(ids)-1)), 'repair must be exact trailing continuation'
    native=combine_singleton_native_inputs([batch.inputs]*len(members),prompt_token_ids=[record['prompt_token_ids']]*len(members),pad_token_id=q.tokenizer.pad_token_id)
    kwargs=exact_history_inputs(q.model,native,full,pad_token_id=q.tokenizer.pad_token_id,logits_to_keep=max(counts)+1)
    with torch.autocast('cuda',dtype=torch.bfloat16):raw=model(**kwargs).logits
    selected=select_compact_replay_logits(raw,counts);losses=[];logical=[]
    for j,seqs,ids,pos,z in zip(members,sequences,full,positions,selected):
        z=z.unsqueeze(0)
        loss,terms=bridge_objective(z,pos,seqs,plan,'local',vocab,j);losses.append(loss)
        logical.append(dict(image_id=image['image_id'],branch='bridge',producer=plan['producer'],raw_identity=record['raw_identity'],tokens=len(ids),visual_tokens=int(__import__('math').prod(record['image_grid_thw'])//4),input_sha256=identity(ids),positions=list(pos),loss=float(loss.detach()),terms={k:float(v.detach()) for k,v in terms.items()},logit_derivatives=diagnostics(terms,z) if diagnostic else {},row_losses=bridge_row_evidence(record,plan,seqs,j),cached_replay=[],logits_sha256=p.tensor_hash(z),bridge=bridge_metadata(plan,'local','bridge',j)))
    loss=torch.stack(losses).sum();lengths=list(map(len,full))
    return loss,dict(image_id=image['image_id'],branch='bridge_group',members=members,logical=logical,loss=float(loss.detach()),logits_shape=list(raw.shape),raw_dtype=str(raw.dtype),selected_dtype=str(selected[0].dtype),shape=dict(batch=len(members),lengths=lengths,left_padding=[max(lengths)-n for n in lengths],padded_tokens=len(members)*max(lengths),unpadded_tokens=sum(lengths),compact_rows=len(members)*(max(counts)+1),selected_rows=sum(counts)),native_sha256={k:p.tensor_hash(kwargs[k]) for k in ('input_ids','attention_mask','position_ids','image_grid_thw')})


def replay_job(q, model, batch, image, record, plan, vocab, job, arm):
    if job['branch']=='bridge_group':return group_forward(q,model,batch,image,record,plan,vocab,job['members'])
    return forward(q,model,batch,image,record,plan,None,vocab,job['branch'],geometry_weight=.1,insertion_policy=arm,branch_index=job.get('branch_index'))


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


def forward(q, model, batch, image, record, plan, encoding, vocab, branch, preservation=None, preservation_weight=0, geometry_weight=0, insertion_policy=None, branch_index=None):
    import torch
    from src.qwen.native import exact_history_inputs
    lineage=None
    original_plan=plan
    if insertion_policy is not None:
        assert branch in ('trace','bridge') and geometry_weight==.1 and preservation_weight==0
        if branch=='trace':plan=bridge_trace_plan(plan,insertion_policy)
    if branch=='bridge':
        sequences=bridge_sequences(image,record,plan,q.tokenizer,branch_index)
        full=sequences[0].input_ids
        positions=tuple(sorted({a.causal_logits_position for seq in sequences for a in seq.atoms}))
    elif branch=='P0':
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
    if branch=='bridge':
        loss,terms=bridge_objective(logits,positions,sequences,plan,insertion_policy,vocab,branch_index)
        rows=bridge_row_evidence(record,plan,sequences,branch_index)
    elif branch=='P0':
        loss,terms=preservation_objective(logits,positions,sequences,vocab,preservation_weight)
        rows=[dict(order=row['order'],atoms=len(seq.atoms)) for row,seq in zip(preservation['rows'],sequences)]
    elif branch=='R':
        loss,terms,rows = retained_credit(logits,sequence,row_ids,vocab,positions)
    elif branch=='redirect':
        loss,terms = redirect_objective(logits,positions,sequence,plan['redirect'],len(record['prompt_token_ids']),vocab)
        rows = []
    else:
        loss,terms = trace_objective(logits,positions,plan,record,image,q.tokenizer,vocab,geometry_weight,original_plan['n'] if insertion_policy is not None else None); rows = []
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
    if insertion_policy is not None:
        if branch=='trace':
            evidence['row_losses']=[dict(order=row['order'],atoms=[a.to_artifact_dict() for a in r.positive_sequence(image,record,row,q.tokenizer).atoms]) for row in plan['M']]
        evidence['bridge']=bridge_metadata(original_plan,insertion_policy,branch,branch_index)
    if lineage is not None:evidence['preservation']=lineage
    if branch=='trace' and geometry_weight:
        evidence['geometry']=dict(weight=geometry_weight,error_slots=list(plan.get('schema_geometry',{}).get('union',erroneous_slots(plan['observations']))))
        if 'schema_geometry' in plan:evidence['schema_geometry']=plan['schema_geometry']
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
    assert updates in (1,2,16,64)
    return tuple(x for x in (0,1,2,4,8,16,32,64) if x<=updates)


def bridge_binding(root, checkpoint, weight, recipe_sha256, arm):
    assert arm in ('local','chain') and weight==.1
    spec=p.load(root/'qualification.json')['bridge']
    assert checkpoint is not None and str(checkpoint)==spec['checkpoint']
    assert recipe_sha256==identity(spec), 'wrong bridge recipe'
    manifest=checkpoint/'inference_payload_manifest.json'
    assert p.digest(manifest)==spec['manifest_sha256']
    payload=p.load(manifest)
    assert payload['schema']=='coordexp-infras-inference-checkpoint-payload-manifest' and payload['schema_version']==1
    for section in ('adapter','special_token_embedding_delta'):
        block=payload[section];assert block['status']=='present'
        for item in block['files']:
            path=checkpoint/block['relative_root']/item['relative_path']
            assert path.stat().st_size==item['size_bytes'] and p.digest(path)==item['sha256']
    return dict(recipe_sha256=recipe_sha256,checkpoint=str(checkpoint),manifest_sha256=spec['manifest_sha256'],weight=weight,insertion_policy=arm)


def geometry_binding(root, checkpoint, weight, recipe_sha256):
    assert weight in (0,.1)
    qualification=p.load(root/'qualification.json')
    assert 'bridge' not in qualification, 'bridge arm required'
    spec=qualification.get('geometry')
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
        geometry_weight=0, start_checkpoint=None, recipe_sha256=None, witness_weight=0, witness_bank_sha256=None, insertion_policy=None, microbatch=1, activation_checkpointing=True, schema_geometry=False):
    import os,time,math,torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from src.qwen.generation import generate_continuations,NativeGenerationPolicy
    from src.losses.vocab import build_token_vocabulary_groups
    from src.artifacts.git_identity import verify_source_identity
    scheduled=export_steps(updates)
    schema_geometry_binding(root,schema_geometry,insertion_policy)
    execution=execution_binding(root,insertion_policy,microbatch,activation_checkpointing)
    geometry=bridge_binding(root,start_checkpoint,geometry_weight,recipe_sha256,insertion_policy) if insertion_policy is not None else geometry_binding(root,start_checkpoint,geometry_weight,recipe_sha256)
    witness, witnesses=witness_binding(root,witness_weight,witness_bank_sha256)
    if witness is not None:assert geometry is not None and geometry_weight==.1 and preservation_weight==0
    if geometry is not None:assert preservation_weight==0 and preservation_bank_sha256 is None
    if insertion_policy is not None:assert updates in (1,2,64) and witness_weight==0 and witness_bank_sha256 is None
    binding,bank=preservation_binding(root,preservation_weight,preservation_bank_sha256)
    rank,out,sources,source = start(output,root); begin=time.monotonic()
    if execution is not None:p.write(out/'execution.json',execution)
    if schema_geometry:p.write(out/'schema-geometry.json',dict(enabled=True))
    if binding is not None:p.write(out/'preservation.json',binding)
    if geometry is not None:p.write(out/'geometry.json',geometry)
    if witness is not None:p.write(out/'witness.json',witness)
    images={x['image_id']:x for x in p.load(RETAINED)}
    inputs={x['image_id']:x for x in p.load(INPUTS)}
    encodings={i:None for i in images} if insertion_policy is not None else {x['image_id']:x for x in p.load(ENCODINGS)}
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
    set_checkpointing(q.model,activation_checkpointing)
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
        if insertion_policy is not None:
            all_plans={x['image_id']:bridge_credit(images[x['image_id']],x,q.tokenizer,producer,insertion_policy,schema_geometry) for x in all_records}
            plans={x['image_id']:all_plans[x['image_id']] for x in records}
        else:plans={x['image_id']:credit(images[x['image_id']],x,q.tokenizer,producer) for x in records}
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
        if witnesses is not None and version in (0,1,4,8,16):
            p.write(out/f'witness-diagnostic-{version}.json',witness_diagnostics(q,batches,witnesses,vocab,producer))
        if version==updates:break
        if insertion_policy is not None:require_bridge_supervision(all_plans,out,version)
        by={x['image_id']:x for x in records};q.model.train();optimizer.zero_grad(set_to_none=True)
        if insertion_policy is not None:
            evidence=r.accumulate_family_step(model,physical_jobs(jobs(list(images),rank,plans,insertion_policy=insertion_policy),microbatch),
                lambda job:replay_job(q,model,batches[job['image_id']],images[job['image_id']],by[job['image_id']],plans[job['image_id']],vocab,job,insertion_policy))
        else:
            evidence=r.accumulate_family_step(model,jobs(list(images),rank,plans,preservation_weight,
                witnesses if witness_weight else (),insertion_policy),
                lambda job:witness_forward(q,model,batches[job['image_id']],witnesses[job['image_id']],vocab,producer,witness_weight)
                if job['branch']=='witness' else forward(q,model,batches[job['image_id']],images[job['image_id']],by[job['image_id']],
                    plans[job['image_id']],encodings[job['image_id']],vocab,job['branch'],
                    bank[job['image_id']] if bank is not None else None,preservation_weight,geometry_weight,insertion_policy,job.get('branch_index')))
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
             geometry_weight=0, start_checkpoint=None, recipe_sha256=None, witness_weight=0, witness_bank_sha256=None, insertion_policy=None, microbatch=1, activation_checkpointing=True, schema_geometry=False):
    """Fresh process, frozen raw shard readback; no model or evaluator truth."""
    schema_geometry_binding(root,schema_geometry,insertion_policy)
    execution=execution_binding(root,insertion_policy,microbatch,activation_checkpointing)
    for path,sha in p.load(root/'qualification.json')['sha256'].items():assert p.digest(path)==sha,path
    binding,bank=preservation_binding(root,preservation_weight,preservation_bank_sha256)
    geometry=bridge_binding(root,start_checkpoint,geometry_weight,recipe_sha256,insertion_policy) if insertion_policy is not None else geometry_binding(root,start_checkpoint,geometry_weight,recipe_sha256)
    witness, witnesses=witness_binding(root,witness_weight,witness_bank_sha256)
    if witness is not None:assert geometry is not None and geometry_weight==.1 and preservation_weight==0
    if geometry is not None:assert preservation_weight==0 and preservation_bank_sha256 is None
    if insertion_policy is not None:assert updates in (1,2,64) and witness_weight==0 and witness_bank_sha256 is None
    inputs={x['image_id']:x for x in p.load(INPUTS)};result=[]
    if insertion_policy is not None:
        tokenizer=r.frontend().tokenizer;images={x['image_id']:x for x in p.load(RETAINED)}
    for rank in range(8):
        directory=output/f'rank-{rank}';receipt=p.load(directory/'complete.json');assert receipt['status']=='complete' and receipt['updates']==updates
        if execution is not None:assert p.load(directory/'execution.json')==execution
        assert (directory/'schema-geometry.json').exists()==schema_geometry
        if schema_geometry:assert p.load(directory/'schema-geometry.json')==dict(enabled=True)
        if binding is not None:assert p.load(directory/'preservation.json')==binding, 'wrong preservation arm'
        if geometry is not None:assert p.load(directory/'geometry.json')==geometry,'wrong geometry arm'
        if witness is not None:assert p.load(directory/'witness.json')==witness,'wrong witness arm'
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
                            assert row['geometry']==dict(weight=geometry_weight,error_slots=[list(x) for x in plans[row['image_id']].get('schema_geometry',{}).get('union',erroneous_slots(plans[row['image_id']]['observations']))])
                            assert row['terms']['Gmax_weighted']==float(__import__('torch').tensor(row['terms']['Gmax_unweighted'],dtype=__import__('torch').float32)*geometry_weight)
            if insertion_policy is not None:
                records={i:p.load(output/f'rollout-{step-1}'/f'rank-{rank}'/f'{i}.json') for i in plans}
                assert all(('schema_geometry' in plan)==schema_geometry for plan in plans.values())
                verify_physical_forwards(evidence['forwards'],list(inputs),rank,plans,records,images,tokenizer,insertion_policy,microbatch)
            if witnesses is not None:
                local=sorted(inputs)[rank::8]
                expected=jobs(list(inputs),rank,plans,0,witnesses if witness_weight else ())
                assert [(x['image_id'],x['branch'],x['sync']) for x in evidence['forwards']]==[(x['image_id'],x['branch'],x['sync']) for x in expected]
                selected=[x for x in evidence['forwards'] if x['branch']=='witness']
                assert [x['image_id'] for x in selected]==([i for i in local if i in witnesses] if witness_weight else [])
                for row in selected:
                    entry=witnesses[row['image_id']]
                    assert row['entry_sha256']==identity(entry) and row['weight']==witness_weight
                    assert row['producer']==evidence['producer']
                    assert row['input_sha256']==entry['input_sha256'] and row['positions']==[entry['causal_position']]
                    assert row['source_producer']==entry['record']['producer'] and row['source_raw_identity']==entry['record']['raw_identity']
                    assert row['loss']==float(__import__('torch').tensor(row['margin'],dtype=__import__('torch').float32)*witness_weight)
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
        if witnesses is not None:
            local=sorted(inputs)[rank::8]
            for version in (v for v in (0,1,4,8,16) if v<=updates):
                diagnostic=p.load(directory/f'witness-diagnostic-{version}.json')
                bound=p.load(directory/f'producer-{version}.json')
                assert diagnostic['producer']==bound['producer'] and diagnostic['parameters_sha256']==identity(bound['parameters'])
                assert diagnostic['unchanged_parameters_and_gradients']
                assert [x['image_id'] for x in diagnostic['rows']]==[i for i in local if i in witnesses]
                for row in diagnostic['rows']:
                    entry=witnesses[row['image_id']]
                    assert row['diagnostic'] and row['weight']==0 and row['entry_sha256']==identity(entry)
                    assert row['input_sha256']==entry['input_sha256'] and row['positions']==[entry['causal_position']]
                    assert row['producer']==bound['producer'] and row['source_producer']==entry['record']['producer']
                    assert row['source_raw_identity']==entry['record']['raw_identity']
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
            geometry_weight=0, start_checkpoint=None, recipe_sha256=None, witness_weight=0, witness_bank_sha256=None, insertion_policy=None):
    """Only this separate process opens evaluator truth, after all raw outputs freeze."""
    for path,sha in p.load(root/'qualification.json')['sha256'].items():assert p.digest(path)==sha,path
    binding,_=preservation_binding(root,preservation_weight,preservation_bank_sha256)
    geometry=bridge_binding(root,start_checkpoint,geometry_weight,recipe_sha256,insertion_policy) if insertion_policy is not None else geometry_binding(root,start_checkpoint,geometry_weight,recipe_sha256)
    witness, witnesses=witness_binding(root,witness_weight,witness_bank_sha256)
    if witness is not None:assert geometry is not None and geometry_weight==.1 and preservation_weight==0
    if geometry is not None:
        assert preservation_weight==0 and preservation_bank_sha256 is None
        for rank in range(8):assert p.load(output/f'rank-{rank}/geometry.json')==geometry
    if witness is not None:
        for rank in range(8):assert p.load(output/f'rank-{rank}/witness.json')==witness
    if binding is not None:
        for rank in range(8):assert p.load(output/f'rank-{rank}/preservation.json')==binding
    if insertion_policy is not None:assert updates in (1,2,64) and witness_weight==0 and witness_bank_sha256 is None
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
    if insertion_policy is not None:
        supply=[];events=0;distinct=set();supervised=set()
        for version in range(updates+1):
            rows=[]
            for rank in range(8):
                for plan in p.load(output/f'rank-{rank}/credit-{version}.json'):
                    selected=plan['bridge']['B'] if plan['bridge'] else []
                    if version<updates:
                        events+=len(selected);distinct.update((plan['image_id'],x['annotation_id']) for x in selected)
                        supervised.update((plan['image_id'],x['annotation_id']) for x in selected+plan['M'])
                    rows.append(dict(image_id=plan['image_id'],m=plan['m'],k=plan['k'],n=plan['n'],eligible=len(plan['eligible']),
                        terminal_insertions=sum(x['boundary']=='terminal' for x in selected),m0=plan['m']==0,
                        successor_multiplicity=bridge_metadata(plan,insertion_policy,'bridge')['successor_multiplicity'],semantic_uncredited_observations=len(plan['observations'])-plan['m'],
                        matched_share=plan['m']/plan['n'] if plan['n'] else 0,insertion_share=plan['k']/plan['n'] if plan['n'] else 0))
            supply.append(dict(version=version,update_follows=version<updates,images=rows,cumulative_insertion_events=events,
                distinct_inserted_ids=sorted(distinct),distinct_supervised_ids=sorted(supervised)))
        p.write(output/'insertion-supply.json',supply)
    for path,sha in p.load(root/'qualification.json')['evaluator_sha256'].items():assert p.digest(path)==sha,path
    partitions=p.load(r.ROOT/'cpu-03/evaluator-partitions.json')
    assert p.digest(r.TRUTH)==partitions['truth_sha256']
    truth=p.load(r.TRUTH)
    scored={k:r.assess_outputs(truth,partitions['hidden10'],v) for k,v in frozen.items()}
    outcomes=r.family_outcomes(scored,[])
    p.write(output/'offline-results.json',outcomes)
    if insertion_policy is not None:
        continuity=[];incoming={}
        for version in range(updates+1):
            for rank in range(8):
                for plan in p.load(output/f'rank-{rank}/credit-{version}.json'):
                    i=plan['image_id']
                    if version==0:incoming[i]=plan
                    for scope,selected in [('current',plan),('incoming',incoming[i])]:
                        if version==updates:continue
                        row=next(x for x in scored[str(version+1)] if x['image_id']==i)
                        targets=[dict(x,kind='M',boundary='original') for x in selected['M']]
                        if selected['bridge']:targets += [dict(x,kind='B') for x in selected['bridge']['B']]
                        for target in targets:
                            continuity.append(dict(scope=scope,selected_version=version if scope=='current' else 0,next_version=version+1,image_id=i,
                                annotation_id=target['annotation_id'],kind=target['kind'],boundary=target['boundary'],m0=selected['m']==0,
                                next_coverage={mode:str(target['annotation_id']) in {str(v) for v in row['ids'][mode]['retained']} for mode in ('raw','category')}))
        p.write(output/'bridge-continuity.json',continuity)
        p.write(output/'stability.json',witness_stability(outcomes,updates,late_start=49))
    if witness is not None:
        p.write(output/'stability.json',witness_stability(outcomes,updates))
        populations=[]
        for version in range(updates+1):
            for rank in range(8):
                for plan in p.load(output/f'rank-{rank}/credit-{version}.json'):
                    rows=plan['observations']
                    for name,values in [('occurrence',rows),('literal_unique',list({x['key']:x for x in rows}.values()))]:
                        for valid in (False,True):
                            selected=[x for x in values if x['valid']==valid]
                            widths=[x['bbox'][2]-x['bbox'][0] for x in selected]
                            heights=[x['bbox'][3]-x['bbox'][1] for x in selected]
                            populations.append(dict(version=version,image_id=plan['image_id'],population=name,valid=valid,
                                widths=widths,heights=heights,widths_le1=sum(x<=1 for x in widths),heights_le1=sum(x<=1 for x in heights)))
        p.write(output/'shape-populations.json',populations)


def witness_stability(outcomes, updates, late_start=9):
    """Post-freeze summaries only; preserve image/version units and acquired IDs."""
    result={}
    burdens=('geometry_invalid','literal_complete_repeats','literal_valid_repeats','malformed','near_repeat_occurrence_pairs','caps')
    for name,versions in [('full',list(range(updates+1))),('late',list(range(late_start,updates+1)))]:
        if not versions:continue
        for mode in ('raw','category'):
            for cohort in ('combined','human13','refined5'):
                rows=[dict(x,version=v) for v in versions for x in outcomes['zero' if v==0 else str(v)]['images']
                      if x['mode']==mode and (cohort=='combined' or x['cohort']==cohort)]
                totals={k:[sum(x['burdens'][k] for x in rows if x['version']==v) for v in versions] for k in burdens}
                acquired={}
                for x in rows:
                    for population in ('retained','hidden'):
                        for ann in x['sets'][population]['gained']:
                            acquired.setdefault(f"{population}:{x['image_id']}:{ann}",[]).append(x['version'])
                result[f'{name}/{mode}/{cohort}']=dict(versions=versions,
                    burdens={k:dict(sum=sum(v),mean=sum(v)/len(v),maximum=max(v)) for k,v in totals.items()},
                    bad_image_versions=sum(any(x['burdens'][k]>0 for k in burdens) for x in rows),
                    bad_versions=sum(any(x['version']==v and any(x['burdens'][k]>0 for k in burdens) for x in rows) for v in versions),
                    empty_image_versions=sum(x['burdens']['valid_rows']==0 for x in rows),acquired_id_versions=acquired)
    return result


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
                    geometry_weight=0, start_checkpoint=None, recipe_sha256=None, witness_weight=0, witness_bank_sha256=None):
    import time,torch
    import torch.distributed as dist
    from src.qwen.native import exact_history_inputs
    from src.losses.vocab import build_token_vocabulary_groups
    from src.artifacts.git_identity import verify_source_identity
    assert preservation_weight==0 and preservation_bank_sha256 is None
    assert witness_weight==0 and witness_bank_sha256 is None
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


def gradient_comparison(current, reference):
    import torch,math
    assert current.keys()==reference.keys()
    rows={};ss=rr=dot=dd=0.;worst=0.
    for name,value in current.items():
        a=value.detach().cpu().double();b=reference[name].double();d=a-b
        av=float((a*a).sum());bv=float((b*b).sum());dv=float((d*d).sum());ab=float((a*b).sum());mx=float(d.abs().max())
        rows[name]=dict(norm=math.sqrt(av),reference_norm=math.sqrt(bv),difference_l2=math.sqrt(dv),relative_l2=math.sqrt(dv/bv) if bv else None,cosine=ab/math.sqrt(av*bv) if av and bv else None,max_abs_difference=mx)
        ss+=av;rr+=bv;dd+=dv;dot+=ab;worst=max(worst,mx)
    return dict(parameters=rows,relative_l2=math.sqrt(dd/rr) if rr else None,cosine=dot/math.sqrt(ss*rr) if ss and rr else None,max_abs_difference=worst)


def replay_benchmark(output, root):
    """Fixed weights, real backward; never generation, optimizer, clipping or save."""
    import time,os,torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from src.losses.vocab import build_token_vocabulary_groups
    from src.artifacts.git_identity import verify_source_identity
    begin=time.monotonic();rank,out,sources,source=start(output,root)
    qual=p.load(root/'qualification.json');spec=qual['benchmark'];profiles=qual['execution']['profiles']
    assert profiles==[dict(arm=a,microbatch=m,activation_checkpointing=c) for a,m,c in [('local',1,True),('local',1,False),('local',4,True),('local',4,False),('chain',1,True),('chain',1,False)]]
    anchor=Path(qual['bridge']['checkpoint']);bridge_binding(root,anchor,.1,identity(qual['bridge']),'local')
    images={x['image_id']:x for x in p.load(RETAINED)};inputs={x['image_id']:x for x in p.load(INPUTS)}
    records={x['image_id']:x for x in frozen_records(Path(spec['trajectories']),list(images),freeze=False)}
    incoming=p.load(spec['producer']);verify_producer(list(records.values()),incoming['producer'],list(images))
    local=sorted(images)[rank::8];dist.init_process_group('nccl')
    q,delta,composition=p.compose(anchor,evaluation=False);p.write(out/'composition.json',composition)
    assert q.model.config.text_config.attention_dropout==0
    assert all(getattr(module,'p',0)==0 for name,module in q.model.named_modules() if 'lora_dropout' in name)
    initial=parameter_identity(q.model);flags={n:x.requires_grad for n,x in q.model.named_parameters()}
    assert initial==incoming['parameters'] and len(initial)==590
    current=dict(kind='fixed_weight_replay_benchmark',source=source['commit'],parameter_sha256=identity(initial))
    p.write(out/'producers.json',dict(source_producer=incoming['producer'],current_producer=current,parameters=initial,flags=flags))
    vocab=build_token_vocabulary_groups(q.token_identity,tokenizer=q.tokenizer)
    batches={i:native_batch(q,inputs[i]) for i in local}
    plans={arm:{i:bridge_credit(images[i],records[i],q.tokenizer,incoming['producer'],arm) for i in local} for arm in ('local','chain')}
    p.write(out/'plans.json',plans)
    set_checkpointing(q.model,True);q.model.train()
    model=DistributedDataParallel(q.model,device_ids=[int(os.environ['LOCAL_RANK'])],broadcast_buffers=False)
    baseline={};receipts=[];setup=time.monotonic()-begin
    for profile_index,profile in enumerate(profiles):
        arm=profile['arm'];microbatch=profile['microbatch'];checkpointing=profile['activation_checkpointing']
        assert execution_binding(root,arm,microbatch,checkpointing)==profile
        set_checkpointing(q.model,checkpointing,enable_inputs=False)
        schedule=physical_jobs(jobs(list(images),rank,plans[arm],insertion_policy=arm),microbatch)
        for pass_index in range(3):
            q.model.zero_grad(set_to_none=True)
            torch.cuda.synchronize();dist.barrier();torch.cuda.reset_peak_memory_stats()
            start_memory=dict(allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved());start_time=time.monotonic()
            evidence=r.accumulate_family_step(model,schedule,lambda job:replay_job(q,model,batches[job['image_id']],images[job['image_id']],records[job['image_id']],plans[arm][job['image_id']],vocab,job,arm))
            torch.cuda.synchronize();elapsed=time.monotonic()-start_time
            peak=dict(allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved());times=[None]*8;dist.all_gather_object(times,elapsed)
            gradients={n:x.grad.detach().cpu().clone() for n,x in q.model.named_parameters() if x.requires_grad}
            assert len(gradients)==590 and all(torch.isfinite(x).all() for x in gradients.values())
            hashes={n:p.tensor_hash(x) for n,x in gradients.items()};synced=[None]*8;dist.all_gather_object(synced,identity(hashes));assert len(set(synced))==1
            assert parameter_identity(q.model)==initial and {n:x.requires_grad for n,x in q.model.named_parameters()}==flags
            logical=logical_forwards(evidence);losses={p.canonical([x['image_id'],x['branch'],x['bridge']['branch_index']]):x['loss'] for x in logical}
            key=(arm,checkpointing);reference=baseline.get(key,baseline.get((arm,True)))
            comparison=None if reference is None else gradient_comparison(gradients,reference['gradients'])
            loss_differences=None if reference is None else {k:dict(loss=v,reference=reference['losses'][k],difference=v-reference['losses'][k]) for k,v in losses.items()}
            receipt=dict(profile=profile,profile_index=profile_index,pass_index=pass_index,measured=pass_index>0,current_producer=current,source_producer=incoming['producer'],forwards=evidence,wall_seconds=elapsed,rank_wall_seconds=times,slowest_rank_seconds=max(times),start_memory=start_memory,peak_memory=peak,gradient_sha256=hashes,synchronized_gradient_hashes=synced,gradient_norms={n:float(x.float().norm()) for n,x in gradients.items()},reference=None if reference is None else reference['identity'],gradient_comparison=comparison,logical_loss_comparison=loss_differences,parameters_and_flags_unchanged=True)
            name=f'profile-{profile_index}-pass-{pass_index}.json';p.write(out/name,receipt);receipts.append(name)
            if microbatch==1 and pass_index==1:baseline[key]=dict(gradients=gradients,losses=losses,identity=dict(profile_index=profile_index,pass_index=pass_index))
    verify_source_identity(source,required_paths=sources)
    p.write(out/'benchmark-complete.json',dict(status='complete',source=source,setup_seconds=setup,wall_seconds=time.monotonic()-begin,profiles=profiles,passes=receipts,reserved_memory_carries_between_profiles=True,optimizer_steps=0,generation_calls=0,artifacts={x.name:p.digest(x) for x in out.glob('*.json')}))
    dist.destroy_process_group()


def replay_benchmark_readback(output, root):
    import math
    qual=p.load(root/'qualification.json');spec=qual['benchmark']
    for path,sha in qual['sha256'].items():assert p.digest(path)==sha,path
    images={x['image_id']:x for x in p.load(RETAINED)};records={x['image_id']:x for x in frozen_records(Path(spec['trajectories']),list(images),freeze=False)}
    incoming=p.load(spec['producer']);tokenizer=r.frontend().tokenizer;summary=[]
    for rank in range(8):
        directory=output/f'rank-{rank}';complete=p.load(directory/'benchmark-complete.json')
        assert complete['status']=='complete' and complete['profiles']==qual['execution']['profiles'] and complete['optimizer_steps']==complete['generation_calls']==0
        assert complete['passes']==[f'profile-{i}-pass-{j}.json' for i in range(6) for j in range(3)]
        for name,sha in complete['artifacts'].items():assert p.digest(directory/name)==sha
        for row in complete['source']['files']:assert p.digest(row['path'])==row['sha256']
        producer=p.load(directory/'producers.json');assert producer['parameters']==incoming['parameters'] and producer['source_producer']==incoming['producer']
        plans={arm:{i:bridge_credit(images[i],records[i],tokenizer,incoming['producer'],arm) for i in sorted(images)[rank::8]} for arm in ('local','chain')}
        assert p.load(directory/'plans.json')==__import__('json').loads(__import__('json').dumps(plans))
        for name in complete['passes']:
            item=p.load(directory/name);profile=item['profile'];assert profile==complete['profiles'][item['profile_index']]
            assert item['parameters_and_flags_unchanged'] and item['current_producer']==producer['current_producer'] and item['source_producer']==incoming['producer']
            verify_physical_forwards(item['forwards'],list(images),rank,plans[profile['arm']],records,images,tokenizer,profile['arm'],profile['microbatch'])
            assert len(item['gradient_sha256'])==len(item['gradient_norms'])==590 and all(math.isfinite(v) for v in item['gradient_norms'].values())
            assert len(set(item['synchronized_gradient_hashes']))==1 and item['synchronized_gradient_hashes'][0]==identity(item['gradient_sha256'])
            logical=logical_forwards(item['forwards']);physical=item['forwards']
            summary.append(dict(rank=rank,profile=profile,pass_index=item['pass_index'],measured=item['measured'],logical_forwards=len(logical),physical_forwards=len(physical),backwards=len(physical),visual_tokens=sum(x['visual_tokens'] for x in logical),unpadded_tokens=sum(x['tokens'] for x in logical),padded_tokens=sum(x['shape']['padded_tokens'] if x['branch']=='bridge_group' else x['tokens'] for x in physical),compact_rows=sum(x['shape']['compact_rows'] if x['branch']=='bridge_group' else len(x['positions']) for x in physical),wall_seconds=item['wall_seconds'],slowest_rank_seconds=item['slowest_rank_seconds'],start_memory=item['start_memory'],peak_memory=item['peak_memory'],gradient_comparison=item['gradient_comparison'],logical_loss_comparison=item['logical_loss_comparison']))
    p.write(output/'benchmark-readback.json',summary)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('run','readback','offline','geometry-replay','replay-benchmark','replay-benchmark-readback'))
    parser.add_argument('--root',type=Path,default=ROOT);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--updates',type=int,choices=(1,2,16,64),default=1)
    parser.add_argument('--preservation-weight',type=float,choices=(0,.25),default=0)
    parser.add_argument('--preservation-bank-sha256')
    parser.add_argument('--geometry-weight',type=float,choices=(0,.1),default=0)
    parser.add_argument('--start-checkpoint',type=Path)
    parser.add_argument('--recipe-sha256')
    parser.add_argument('--witness-weight',type=float,choices=(0,.1),default=0)
    parser.add_argument('--witness-bank-sha256')
    parser.add_argument('--insertion-policy',choices=('local','chain'))
    parser.add_argument('--schema-geometry',action='store_true')
    parser.add_argument('--microbatch',type=int,choices=(1,4),default=1)
    parser.add_argument('--activation-checkpointing',choices=('on','off'),default='on')
    a=parser.parse_args()
    if a.command in ('replay-benchmark','replay-benchmark-readback'):
        assert not a.schema_geometry and a.microbatch==1 and a.activation_checkpointing=='on' and a.insertion_policy is None and a.start_checkpoint is None and a.recipe_sha256 is None
        assert a.updates==1 and a.preservation_weight==a.geometry_weight==a.witness_weight==0 and a.preservation_bank_sha256 is None and a.witness_bank_sha256 is None
        return {'replay-benchmark':replay_benchmark,'replay-benchmark-readback':replay_benchmark_readback}[a.command](a.output,a.root)
    extra={'insertion_policy':a.insertion_policy} if a.command!='geometry-replay' else {}
    if a.command in ('run','readback'):extra.update(microbatch=a.microbatch,activation_checkpointing=a.activation_checkpointing=='on',schema_geometry=a.schema_geometry)
    else:assert not a.schema_geometry and a.microbatch==1 and a.activation_checkpointing=='on'
    assert a.command!='geometry-replay' or a.insertion_policy is None
    {'run':run,'readback':readback,'offline':offline,'geometry-replay':geometry_replay}[a.command](
        a.output,a.root,a.updates,a.preservation_weight,a.preservation_bank_sha256,a.geometry_weight,a.start_checkpoint,a.recipe_sha256,a.witness_weight,a.witness_bank_sha256,**extra)


if __name__=='__main__':main()
