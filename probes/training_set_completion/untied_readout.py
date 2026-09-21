"""Bounded native readout event selection and independently runnable CPU reduction."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def token_hash(tokens):
    return hashlib.sha256(json.dumps(list(tokens), separators=(',', ':')).encode()).hexdigest()


def select_events(source_order, trajectories):
    """Consume authoritative row candidates, without upgrading proxies to physical owners.

    trajectories[(image_key, model)] requires token_ids, rows, identity; each row
    has row_index, positions ({offset, role}), recurrence_candidate and evidence.
    Both original model trajectories must exist before an image can be admitted.
    """
    selected = []
    images = []
    for image in source_order:
        pair = [trajectories[(image, model)] for model in ('tied', 'untied')]
        if not any(any(r['recurrence_candidate'] for r in t['rows']) for t in pair):
            continue
        images.append(image)
        for model, trajectory in zip(('tied', 'untied'), pair):
            rows = trajectory['rows']
            recurrence = [r for r in rows if r['recurrence_candidate']]
            chosen = {r['row_index']: r for r in recurrence[:2] + recurrence[-1:]}
            earlier = [r for r in rows if not r['recurrence_candidate'] and
                       (not recurrence or r['row_index'] < recurrence[0]['row_index'])]
            if earlier:
                chosen[earlier[-1]['row_index']] = earlier[-1]
            for index, row in sorted(chosen.items()):
                positions = row['positions']
                if len(positions) > 10 or len({p['offset'] for p in positions}) != len(positions):
                    raise ValueError('event positions must be unique and at most ten per row')
                for position in positions:
                    offset = position['offset']
                    if not 0 <= offset < len(trajectory['token_ids']):
                        raise ValueError('event offset outside native trajectory')
                    selected.append(dict(image_key=image, model=model, row_index=index,
                        **position, prefix_sha256=token_hash(trajectory['token_ids'][:offset]),
                        selected_token=trajectory['token_ids'][offset], identity=trajectory['identity'],
                        recurrence_candidate=row['recurrence_candidate'],
                        recurrence_evidence=row.get('evidence'),
                        physical_status=row.get('physical_status', 'HOLD')))
        if len(images) == 12:
            break
    return {'image_keys': images, 'events': selected, 'status': 'frozen_cpu',
            'selection': 'first12 qualifying in frozen source order; onset,next,last plus earlier healthy'}


def affine_accounting(logits):
    curve = logits.detach().double().cpu()
    if curve.shape != (1000,) or not torch.isfinite(curve).all():
        raise ValueError('expected finite 1000-bin curve')
    x = torch.arange(1000, dtype=torch.float64) / 999
    slope = ((x-x.mean()) * (curve-curve.mean())).sum() / ((x-x.mean())**2).sum()
    intercept = curve.mean() - slope*x.mean()
    residual = curve - intercept - slope*x
    interior = int(curve[1:-1].argmax()) + 1
    margins = {}
    for endpoint in (0, 999):
        linear = slope * (x[endpoint]-x[interior])
        remainder = residual[endpoint]-residual[interior]
        observed = curve[endpoint]-curve[interior]
        margins[str(endpoint)] = {'best_interior': interior, 'margin': float(observed),
            'slope_contribution': float(linear), 'residual_contribution': float(remainder),
            'reconstruction_error': float(abs(observed-linear-remainder))}
    return {'intercept': float(intercept), 'slope': float(slope),
            'residual': residual, 'second_difference': curve[2:]-2*curve[1:-1]+curve[:-2],
            'endpoint_margins': margins}


def reduce_capture(packet):
    """Saved tensors only: no model load, no GPU and no normalized-state substitution."""
    weights = packet['coordinate_weights'].double().cpu()
    hidden = packet['post_norm'].double().cpu()
    pre = packet['pre_norm'].float().cpu()
    gain = packet['norm_gain'].float().cpu()
    logits = packet['logits'].double().cpu()
    ids = packet['coordinate_ids'].long().cpu()
    if weights.shape != (1000, hidden.numel()) or ids.shape != (1000,):
        raise ValueError('coordinate row shape mismatch')
    reconstructed_norm = pre * torch.rsqrt(pre.square().mean() + packet['norm_eps']) * gain
    norms = weights.norm(dim=1)
    if (norms == 0).any() or hidden.norm() == 0:
        raise ValueError('angular alignment undefined for zero vectors')
    alignment = (weights @ hidden) / (norms * hidden.norm())
    curve = logits[ids]
    fit = affine_accounting(curve)
    policy_norms = packet['coordinate_weights'].float().cpu().double().norm(dim=1)
    factors = policy_norms.median() / policy_norms  # Frozen predecessor uses lower central median.
    normalized = (curve * factors).to(packet['logits'].dtype)
    chosen = int(packet['selected_token'])
    competitors = {str(k): float(logits[chosen]-logits[int(k)]) for k in packet['competitors']}
    return {**fit, 'row_norms': norms, 'angular_alignment': alignment,
        'coordinate_reconstruction_max_abs': float((norms*hidden.norm()*alignment-curve).abs().max()),
        'norm_reconstruction_max_abs': float((reconstructed_norm.double()-hidden).abs().max()),
        'selected_rank': 1+int((logits > logits[chosen]).sum()), 'competitor_margins': competitors,
        'full_vocab_logsumexp': float(logits.logsumexp(0)),
        'immediate_readout_counterfactual_coordinate_logits': normalized,
        'counterfactual_scope': 'same native hidden state; no trajectory or causal layer claim'}


def self_check():
    x = torch.arange(1000, dtype=torch.float64)/999
    curve = 2+3*x
    curve[0] += 4
    report = affine_accounting(curve)
    assert all(m['reconstruction_error'] < 1e-12 for m in report['endpoint_margins'].values())
    assert report['endpoint_margins']['0']['margin'] > 0
    assert report['endpoint_margins']['0']['slope_contribution'] < 0
    trajectory = {'token_ids': list(range(20)), 'identity': 'synthetic', 'rows': [
        {'row_index': i, 'positions': [{'offset': i, 'role': 'x1'}],
         'recurrence_candidate': i in (2,3,4,5)} for i in range(6)]}
    trajectories = {(str(i), m): trajectory for i in range(14) for m in ('tied','untied')}
    events = select_events(list(map(str, range(14))), trajectories)
    assert len(events['image_keys']) == 12
    assert {e['row_index'] for e in events['events']} == {1,2,3,5}
    assert all(e['physical_status'] == 'HOLD' for e in events['events'])
    generator = torch.Generator().manual_seed(19)
    weights = torch.randn(1000, 8, generator=generator)
    pre = torch.randn(8, generator=generator)
    gain = torch.linspace(.5, 1.5, 8)
    hidden = pre * torch.rsqrt(pre.square().mean()+1e-6) * gain
    logits = weights @ hidden
    packet = dict(coordinate_weights=weights, pre_norm=pre, post_norm=hidden,
                  norm_gain=gain, norm_eps=1e-6, logits=logits, coordinate_ids=torch.arange(1000),
                  selected_token=int(logits.argmax()), competitors=[0,999])
    reduced = reduce_capture(packet)
    assert reduced['norm_reconstruction_max_abs'] == 0
    assert reduced['coordinate_reconstruction_max_abs'] < 2e-6
    packet['post_norm'] = hidden + .1
    assert reduce_capture(packet)['norm_reconstruction_max_abs'] > .09
    print('PASS affine exact-margin reconstruction and frozen first12 onset/next/last selection')


def static_audit(base, checkpoints, output):
    from safetensors import safe_open
    output.mkdir(parents=True, exist_ok=False)
    weight_map = json.loads((base / 'model.safetensors.index.json').read_text())['weight_map']
    key = 'model.language_model.embed_tokens.weight'
    with safe_open(base / weight_map[key], framework='pt', device='cpu') as handle:
        base_rows = handle.get_slice(key)[151670:152670].float()
    key = 'model.language_model.norm.weight'
    with safe_open(base / weight_map[key], framework='pt', device='cpu') as handle:
        gain = handle.get_tensor(key).float()
    tensors = {'base_coordinate_rows': base_rows, 'norm_gain': gain}
    report = {'status': 'static_candidate', 'model_forwards': 0, 'gpu_seconds': 0, 'models': {}}
    for name, checkpoint in checkpoints.items():
        folder = checkpoint / 'special_token_embeddings'
        metadata = json.loads((folder / 'special_token_embeddings.json').read_text())
        positions = torch.tensor([metadata['token_ids'].index(i) for i in range(151670,152670)])
        payload = folder / 'special_token_embeddings.safetensors'
        models = {}
        with safe_open(payload, framework='pt', device='cpu') as handle:
            for direction in ('input', 'output'):
                key = 'shared_embed_delta' if name == 'tied' else direction + '_embed_delta'
                delta = handle.get_tensor(key).float()[positions]
                effective = base_rows + delta
                norms = effective.norm(dim=1)
                tensors[name+'_'+direction+'_delta'] = delta
                tensors[name+'_'+direction+'_effective'] = effective
                centered = effective.double()-effective.double().mean(0)
                singular = torch.linalg.svdvals(centered)
                energy = singular.square()
                models[direction] = {
                    'endpoint_norm_ranks': {str(i): 1+int((norms > norms[i]).sum()) for i in (0,999)},
                    'endpoint_norms': {str(i): float(norms[i]) for i in (0,999)},
                    'norm_median_fp64': float(torch.quantile(norms.double(), .5)),
                    'adjacent_cosine_mean': float(torch.nn.functional.cosine_similarity(effective[:-1],effective[1:]).mean()),
                    'centered_energy_top1': float(energy[:1].sum()/energy.sum()),
                    'centered_energy_top10': float(energy[:10].sum()/energy.sum()),
                    'effective_sha256': hashlib.sha256(effective.numpy().tobytes()).hexdigest()}
        report['models'][name] = {'checkpoint': str(checkpoint),
            'payload_sha256': hashlib.sha256(payload.read_bytes()).hexdigest(), **models}
    torch.save(tensors, output / 'static-weights.pt')
    report['tensor_bytes'] = sum(t.numel()*t.element_size() for t in tensors.values())
    report['weights_sha256'] = hashlib.sha256((output/'static-weights.pt').read_bytes()).hexdigest()
    (output/'static.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


def freeze_native_events(root, output):
    from src.inference.parsing import parse_compact_object_box_closed
    from probes.training_set_completion.paired_evaluation import _matchable_rows_with_geometry_debt
    from probes.training_set_completion.source256_evaluation import _strict_repeat_rows
    from probes.training_set_completion.readout_norm_fresh import _binding
    panel = json.loads((root/'panel.json').read_text())
    trajectories = {}
    sources = []
    order_path=root/'event-selection-order.json'
    source_order=json.loads(order_path.read_text())['image_ids']
    if len(source_order)!=len(set(source_order)):raise ValueError('event source order repeats an identity')
    inspected=[]
    qualifying=0
    for image in source_order:
        group = next(g for g in panel['groups'] if any(int(c['input_record']['image_id']) == image for c in g['cases']))
        case = next(c for c in group['cases'] if int(c['input_record']['image_id']) == image)
        for model in ('tied','untied'):
            folder = root/'runtime'/f'{model}-original'/group['key']
            if not (folder/'receipt.json').exists():
                raise ValueError(f'NEEDS_CONTEXT next image={image} group={group["key"]} model={model}; qualifying={qualifying}; inspected={len(inspected)}')
            receipt = json.loads((folder/'receipt.json').read_text())
            if receipt['status'] != 'candidate_complete':
                raise ValueError(f'NEEDS_CONTEXT next image={image} group={group["key"]} model={model}; qualifying={qualifying}; inspected={len(inspected)}')
            raw_path = folder/'raw.json'
            raw = json.loads(raw_path.read_text())
            trace_path = folder/'trace.json'
            if receipt['raw'] != _binding(raw_path) or receipt['trace'] != _binding(trace_path):
                raise ValueError('native trajectory binding mismatch')
            saved = next(r for r in raw['rows'] if r['image_id'] == image)
            parsed = parse_compact_object_box_closed(saved['text'], **{k:case[k] for k in ('row_id','row_index','image_width','image_height')}).to_artifact_dict()
            valid, _ = _matchable_rows_with_geometry_debt({**parsed,'pred':parsed['predictions']})
            repeats = {r['generated_order']:r for r in _strict_repeat_rows(valid)}
            tokens = saved['token_ids']
            starts = [i for i,t in enumerate(tokens) if t == 151646]
            rows = []
            for row in valid:
                order = row['generated_order']; start = starts[order]
                end = starts[order+1] if order+1 < len(starts) else len(tokens)
                coord_positions = [i for i in range(start,end) if 151670 <= tokens[i] < 152670]
                if len(coord_positions) != 4:
                    raise ValueError('valid parser row does not bind four literal coordinate tokens')
                if [tokens[i]-151670 for i in coord_positions] != row['coord_bins_1000']:
                    raise ValueError('parser/token geometry mismatch')
                positions = [{'offset':start,'role':'object_start'}]
                positions += [{'offset':i,'role':role} for i,role in zip(coord_positions,('x1','y1','x2','y2'))]
                positions += [{'offset':i,'role':'row_end' if tokens[i]==151649 else 'eos'} for i in range(start,end) if tokens[i] in (151649,151645)]
                rows.append(dict(row_index=order,positions=positions,recurrence_candidate=order in repeats,evidence=repeats.get(order),physical_status='HOLD'))
            trajectories[(image,model)] = dict(token_ids=tokens,rows=rows,identity=dict(group=group['key'],native_receipt=_binding(folder/'receipt.json'),raw=_binding(raw_path),trace=_binding(trace_path)))
            sources.append(_binding(raw_path))
        inspected.append(image)
        qualifying+=int(any(any(r['recurrence_candidate'] for r in trajectories[(image,m)]['rows']) for m in ('tied','untied')))
        if qualifying==12:break
    frozen = select_events(inspected,trajectories)
    frozen['source_order']=_binding(order_path)
    frozen['inspected_source_prefix']=inspected
    frozen.update(panel=_binding(root/'panel.json'),sources=sources,qualification='class-agnostic later valid row IoU strictly > .95; physical HOLD')
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as handle:
        json.dump(frozen,handle,indent=2)
    return frozen


def capture_native(root, manifest_path, output, model_key, device):
    """One full original batch-prefix replay per selected offset; fail closed on parity."""
    import time
    import os
    from transformers import GenerationConfig
    from probes.training_set_completion.untied_shared import load_model
    from probes.training_set_completion.readout_norm_fresh import _binding, _input_identity
    from src.inference.bound_requests import build_bound_native_requests
    from src.qwen.native import prepare_native_inputs, _STALE_HISTORY_FIELDS
    gate = json.loads((root/'shared-gate.json').read_text())
    if gate.get('status') not in ('passed','candidate_complete','admitted'):
        raise ValueError('shared gate not admitted')
    panel = json.loads((root/'panel.json').read_text())
    manifest = json.loads(manifest_path.read_text())
    if manifest['panel'] != _binding(root/'panel.json') or manifest['status'] != 'frozen_cpu':
        raise ValueError('event manifest/panel mismatch')
    events = [e for e in manifest['events'] if e['model']==model_key]
    output.mkdir(parents=True,exist_ok=False)
    ledger = dict(status='running',pid=os.getpid(),model=model_key,model_forwards=0,vision_forwards=0,
                  gpu_seconds=0,tensor_bytes=0,manifest=_binding(manifest_path),captures=[])
    def persist():
        (output/'receipt.json').write_text(json.dumps(ledger,indent=2)+'\n')
    persist(); started=time.monotonic(); handles=[]
    try:
        q,identity = load_model(model_key,device); m=q.model; h=m.get_output_embeddings(); language=m.model.language_model
        ledger['identity']=identity
        selected_ids=h.selected_token_ids
        weights=(h.base.weight[selected_ids]+h.shared_embed_delta).detach().cpu()[4:]
        coordinates=selected_ids.detach().cpu()[4:]
        gain=language.norm.weight.detach().cpu(); eps=language.norm.variance_epsilon
        torch.save(dict(coordinate_weights=weights,coordinate_ids=coordinates,norm_gain=gain,norm_eps=eps),output/'weights.pt')
        ledger['tensor_bytes']=(output/'weights.pt').stat().st_size
        active_indices=[]; captured={}
        def count(*args):
            ledger['model_forwards']+=1
            if ledger['model_forwards']>1000 or time.monotonic()-started>3*3600:
                raise RuntimeError('per-model half of unit forward/time budget exhausted')
        def vision(*args):ledger['vision_forwards']+=1
        def store(name, value):
            if isinstance(value,tuple):value=value[0]
            captured[name]=value[active_indices,-1,:].detach().cpu().clone()
        handles=[m.register_forward_pre_hook(count),m.model.visual.register_forward_pre_hook(vision),
                 language.norm.register_forward_pre_hook(lambda mod,args:store('pre_norm',args[0])),
                 h.register_forward_pre_hook(lambda mod,args:store('post_norm',args[0])),
                 h.register_forward_hook(lambda mod,args,out:store('logits',out))]
        for index,layer in enumerate(language.layers):
            handles.extend([layer.register_forward_pre_hook(lambda mod,args,i=index:store(f'layer{i}.residual',args[0])),
                layer.self_attn.register_forward_hook(lambda mod,args,out,i=index:store(f'layer{i}.attention',out)),
                layer.mlp.register_forward_hook(lambda mod,args,out,i=index:store(f'layer{i}.mlp',out))])
        for group_key in dict.fromkeys(e['identity']['group'] for e in events):
            group=next(g for g in panel['groups'] if g['key']==group_key)
            folder=root/'runtime'/f'{model_key}-original'/group_key
            native_receipt=json.loads((folder/'receipt.json').read_text())
            for field in ('input_rows_sha256','output_rows_sha256','input_delta_sha256','output_delta_sha256'):
                if identity[field] != native_receipt['identity'][field]:raise ValueError('loaded native model identity mismatch')
            raw=json.loads((folder/'raw.json').read_text())['rows']; trace=json.loads((folder/'trace.json').read_text())['steps']
            if [r['image_id'] for r in raw] != [int(c['input_record']['image_id']) for c in group['cases']]:raise ValueError('native companion order mismatch')
            group_events=[e for e in events if e['identity']['group']==group_key]
            for event in group_events:
                for key in ('raw','trace','native_receipt'):
                    binding=event['identity'][key]
                    if binding!=_binding(Path(binding['path'])):raise ValueError('native source changed')
            config=dict(panel['configs'][model_key]);config['data']=dict(input_jsonl=group['input_jsonl'])
            requests,_=build_bound_native_requests(q,config,group['cases'])
            batch=prepare_native_inputs(q.processor,requests,device=device,record_media_identity=True)
            if _input_identity(batch)!=native_receipt['input_identity']:raise ValueError('native batch identity mismatch')
            for offset in sorted({e['offset'] for e in group_events}):
                current=[e for e in group_events if e['offset']==offset]
                active_indices[:]=[next(i for i,r in enumerate(raw) if r['image_id']==e['image_key']) for e in current]
                for event,bi in zip(current,active_indices):
                    if token_hash(raw[bi]['token_ids'][:offset])!=event['prefix_sha256']:raise ValueError('prefix mismatch')
                suffix=[r['token_ids'][:offset]+[q.tokenizer.pad_token_id]*max(0,offset-len(r['token_ids'])) for r in raw]
                inputs={k:v for k,v in batch.inputs.items() if k not in _STALE_HISTORY_FIELDS and k not in ('use_cache','return_dict','logits_to_keep')}
                inputs['input_ids']=torch.cat((batch.inputs['input_ids'],torch.tensor(suffix,device=device,dtype=torch.long)),dim=1)
                inputs['attention_mask']=torch.cat((batch.inputs['attention_mask'],torch.ones(len(raw),offset,device=device,dtype=batch.inputs['attention_mask'].dtype)),dim=1)
                captured.clear()
                with torch.inference_mode():
                    result=m.generate(**inputs,generation_config=GenerationConfig(max_new_tokens=1,do_sample=False,repetition_penalty=1,eos_token_id=151645,pad_token_id=q.tokenizer.pad_token_id),use_model_defaults=False)
                for j,(event,bi) in enumerate(zip(current,active_indices)):
                    packet={name:value[j].clone() for name,value in captured.items()}
                    packet.update(coordinate_weights=weights,coordinate_ids=coordinates,norm_gain=gain,norm_eps=eps,
                                  selected_token=event['selected_token'],competitors=[0,151645,151646,151649,int(trace[offset]['raw_runnerups'][bi])])
                    logits=packet['logits'];top=logits.topk(2)
                    error=max(abs(float(top.values[k])-trace[offset]['raw_top2'][bi][k]) for k in (0,1))
                    native_margin=trace[offset]['raw_top2'][bi][0]-trace[offset]['raw_top2'][bi][1]
                    parity=int(logits.argmax())==event['selected_token'] and error<=panel['tolerances']['logit_atol']
                    name=f'{event["image_key"]}-{event["row_index"]}-{offset}.pt'
                    # Store shared weight once; reducers merge it on demand.
                    saved={k:v for k,v in packet.items() if k not in ('coordinate_weights','coordinate_ids','norm_gain','norm_eps')}
                    path=output/name;torch.save(saved,path)
                    reduction=reduce_capture(packet)
                    torch.save(reduction,output/(name+'.reduced.pt'))
                    ledger['tensor_bytes']+=path.stat().st_size+(output/(name+'.reduced.pt')).stat().st_size
                    ledger['captures'].append(dict(event=event,tensors=_binding(path),replay_parity=parity,top2_max_abs_error=error,native_margin=native_margin,
                       rank_change_claim_admissible=parity and 2*error<native_margin,
                       coordinate_reconstruction_max_abs=reduction['coordinate_reconstruction_max_abs'],norm_reconstruction_max_abs=reduction['norm_reconstruction_max_abs']))
                    if ledger['tensor_bytes']>8*1024**3:raise RuntimeError('per-model tensor limit exhausted')
                    persist()
                    if not parity:raise RuntimeError('native replay parity failed; no further events admitted')
        ledger['status']='candidate_complete'
    except BaseException as exc:
        ledger.update(status='partial_candidate',error=repr(exc));raise
    finally:
        for handle in handles:handle.remove()
        ledger['gpu_seconds']=time.monotonic()-started;ledger['live_jobs']=[];persist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-check', action='store_true')
    parser.add_argument('--capture', type=Path)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--freeze-events', action='store_true')
    parser.add_argument('--native-events', type=Path)
    parser.add_argument('--model', choices=['tied','untied'])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--weights', type=Path)
    parser.add_argument('--static-base', type=Path)
    parser.add_argument('--tied-checkpoint', type=Path)
    parser.add_argument('--untied-checkpoint', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.self_check:
        self_check()
    elif args.freeze_events and args.root and args.output:
        freeze_native_events(args.root,args.output)
    elif args.native_events and args.root and args.output and args.model:
        capture_native(args.root,args.native_events,args.output,args.model,args.device)
    elif args.static_base and args.tied_checkpoint and args.untied_checkpoint and args.output:
        torch.set_num_threads(4)
        static_audit(args.static_base, {'tied': args.tied_checkpoint, 'untied': args.untied_checkpoint}, args.output)
    elif args.capture and args.output:
        packet=torch.load(args.capture, map_location='cpu', weights_only=True)
        if args.weights: packet.update(torch.load(args.weights, map_location='cpu', weights_only=True))
        torch.save(reduce_capture(packet), args.output)
    else:
        parser.error('supply --self-check or --capture and --output')


if __name__ == '__main__':
    main()
