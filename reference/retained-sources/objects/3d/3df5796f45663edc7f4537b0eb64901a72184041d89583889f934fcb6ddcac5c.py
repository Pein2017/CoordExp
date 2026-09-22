"""Fixed18 selected-delta backward accounting; no optimizer or parameter updates."""
import argparse
import gc
import hashlib
import json
import os
import signal
import time
from pathlib import Path

import torch

from probes.training_set_completion.training import masked_ce_loss, raw_axis_validity_hinge
from src.inference.bound_requests import build_bound_native_requests
from src.qwen.native import prepare_native_inputs, prepare_replay

ROOT = Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-untied-gradient-path-accounting')
TOL = dict(forward_max_abs=1e-6, gradient_max_abs=5e-5, gradient_relative_l2=1e-4,
           fd_epsilon=.01, fd_atol=2e-5, fd_rtol=.05)


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def thash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def residual(a, b):
    return dict(max_abs=float((a-b).abs().max()), relative_l2=float((a-b).norm()/a.norm().clamp_min(1e-20)))


def prepare(panel_path, routes_path):
    from src.qwen.runtime_loading import QwenLoadOptions, load_qwen_components_from_options
    panel=json.loads(panel_path.read_text())
    qwen=load_qwen_components_from_options(QwenLoadOptions(base_model=panel['configs']['tied']['model']['base_model'],dtype='fp32',attn_implementation='sdpa',load_model=False))
    tok=qwen.tokenizer;coords=[tok.convert_tokens_to_ids(f'<|coord_{i}|>') for i in range(1000)]
    box_start=tok.convert_tokens_to_ids('<|box_start|>');box_end=tok.convert_tokens_to_ids('<|box_end|>')
    routes=[]
    for iid in panel['refined_source_order']:
        teacher=panel['teachers'][str(iid)];tokens=teacher['token_ids'];case=teacher['case']
        assert tokens[-1]==151645
        positions=[i for i,t in enumerate(tokens) if t==box_start]
        assert len(positions)==len(teacher['objects'])
        boxes=[]
        for pos,obj in zip(positions,teacher['objects']):
            bins=[coords.index(t) for t in tokens[pos+1:pos+5]]
            assert tokens[pos+5]==box_end and [f'<|coord_{v}|>' for v in bins]==obj['bbox_2d']
            boxes.append(dict(zip(['x1_position','y1_position','x2_position','y2_position'],range(pos+1,pos+5)),expected_bins=bins))
        requests,_=build_bound_native_requests(qwen,panel['configs']['tied'],[case])
        batch=prepare_native_inputs(qwen.processor,requests,device='cpu',record_media_identity=True)
        routes.append(dict(image_id=iid,case=case,prompt_token_ids=list(batch.prompt_token_ids[0]),continuation_token_ids=tokens,
            ce_weights=[1]*len(tokens),trusted_boxes=boxes,image_identity=dict(executed_media_sha256=batch.media_sha256[0],observed_image_grid_thw=list(batch.image_grids[0])),
            provenance=dict(panel_sha256=hashlib.sha256(panel_path.read_bytes()).hexdigest(),teacher_image_id=iid)))
    assert len(routes)==18 and sum(len(r['trusted_boxes']) for r in routes)==570
    assert not routes_path.exists();routes_path.parent.mkdir(parents=True,exist_ok=True)
    write(routes_path,dict(routes=routes,complete_boxes=570,incomplete_boxes=0,tolerances=TOL,planned_evaluations=201))


def describe(g, w, coordinate_rows):
    norms = w.norm(dim=1)
    radial = (g*w).sum(dim=1)/norms.clamp_min(1e-20)
    coord = radial[coordinate_rows]
    interior = coord[1:999]
    wrappers = [i for i in range(len(w)) if i not in set(coordinate_rows)]
    return dict(norm=float(g.norm()), row_gradient_norms=g.norm(dim=1).tolist(),
                radial=radial.tolist(), zero_weight_rows=(norms==0).nonzero().flatten().tolist(),
                plain_gd_norm_change=(-radial).tolist(),
                coord0=float(coord[0]), coord999=float(coord[999]),
                interior_quantiles=torch.quantile(interior, torch.tensor([0., .25, .5, .75, 1.])).tolist(),
                endpoint_interior_percentiles=[float((interior<=coord[i]).float().mean()) for i in [0,999]],
                wrapper_rows=wrappers, wrapper_radial=radial[wrappers].tolist())


def reduce(root):
    """CPU reconstruction from raw tensors, with equal-scene global reduction."""
    result = dict(conditions={}, tied_sum={})
    aggregate = {}; scene_sets={}
    for condition in ['tied-shared', 'tied-split', 'untied']:
        files = sorted((root/condition).glob('scene-*.pt'))
        assert len(files)==18, (condition, len(files))
        meta = torch.load(root/condition/'weights.pt', map_location='cpu', weights_only=True)
        scenes = [torch.load(p, map_location='cpu', weights_only=True) for p in files]
        scene_sets[condition]=scenes
        assert len({s['image_id'] for s in scenes})==18
        assert sum(s['complete_boxes'] for s in scenes)==570
        paths = scenes[0]['gradients']['ce'].keys()
        gradients = {o: {p: sum((s['gradients'][o][p] for s in scenes), torch.zeros_like(scenes[0]['gradients'][o][p]))/18 for p in paths} for o in ['ce','axis']}
        gradients['weighted_axis'] = {p:g*.01 for p,g in gradients['axis'].items()}
        gradients['combined'] = {p:gradients['ce'][p]+gradients['weighted_axis'][p] for p in paths}
        aggregate[condition] = gradients
        tables = {}
        for objective, values in gradients.items():
            table = {p:describe(g, meta['weights'][p], meta['coordinate_rows']) for p,g in values.items()}
            if set(values)=={'in','out'}:
                a,b=values['in'].flatten(),values['out'].flatten(); na,nb=float(a.norm()),float(b.norm())
                table['in_out']=dict(norm_ratio=na/nb if nb else None, cosine=float(torch.dot(a,b)/(na*nb)) if na and nb else None, zero_in=na==0,zero_out=nb==0)
            tables[objective]=table
        result['conditions'][condition]=dict(scene_count=18,complete_boxes=570,incomplete_boxes=sum(s['incomplete_boxes'] for s in scenes),
            active_tokens=sum(s['active_tokens'] for s in scenes), losses={o:sum(s['losses'][o] for s in scenes)/18 for o in ['ce','axis']},tables=tables)
    for objective in ['ce','axis']:
        a=aggregate['tied-shared'][objective]['shared']; b=aggregate['tied-split'][objective]
        result['tied_sum'][objective]=residual(a,b['in']+b['out'])
        assert result['tied_sum'][objective]['max_abs']<=TOL['gradient_max_abs'] and result['tied_sum'][objective]['relative_l2']<=TOL['gradient_relative_l2']
    result['per_scene_tied_sum']=[]
    for a,b,c in zip(*(scene_sets[k] for k in ['tied-shared','tied-split','untied'])):
        for key in ['image_id','target_sha256','mask_sha256','complete_boxes','active_tokens']:
            assert a[key]==b[key]==c[key],key
        checks={o:residual(a['gradients'][o]['shared'],b['gradients'][o]['in']+b['gradients'][o]['out']) for o in ['ce','axis']}
        assert all(r['max_abs']<=TOL['gradient_max_abs'] and r['relative_l2']<=TOL['gradient_relative_l2'] for r in checks.values())
        result['per_scene_tied_sum'].append(dict(image_id=a['image_id'],checks=checks))
    torch.save(aggregate,root/'aggregate.pt')
    write(root/'summary.json',result)
    return result


def execute(panel_path, routes_path, gate_path, root):
    from probes.training_set_completion.untied_shared import load_model
    assert gate_path.is_file(), 'shared loader admission receipt required'
    assert json.loads(gate_path.read_text())['status']=='passed', 'shared loader admission failed'
    panel=json.loads(panel_path.read_text()); routes=json.loads(routes_path.read_text())['routes']
    assert len(routes)==18 and sum(len(r['trusted_boxes']) for r in routes)==570
    assert [r['image_id'] for r in routes]==panel['refined_source_order']
    panel_hash=hashlib.sha256(panel_path.read_bytes()).hexdigest()
    assert all(r['provenance']['panel_sha256']==panel_hash and r['continuation_token_ids']==panel['teachers'][str(r['image_id'])]['token_ids'] for r in routes)
    assert all(r['continuation_token_ids'][-1]==151645 and r['ce_weights'][-1]==1 for r in routes)
    root.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(17); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    start=time.monotonic()
    receipt=dict(status='running',pid=os.getpid(),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),tolerances=TOL,
        precision='FP32, SDPA, no autocast, no TF32',seed=17,scene_order=[r['image_id'] for r in routes],chunk_sizes=[1]*18,
        global_reduction='per-scene gradient sum / 18; CE active-token mean; axis complete-box mean retaining zero-box scenes',
        panel_sha256=hashlib.sha256(panel_path.read_bytes()).hexdigest(),routes_sha256=hashlib.sha256(routes_path.read_bytes()).hexdigest(),
        shared_gate_sha256=hashlib.sha256(gate_path.read_bytes()).hexdigest(),producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        counts=dict(forwards=0,backwards=0),gates={},complete_passes=0,limits=dict(evaluations=300,gpu_hours=6,retained_bytes=4*1024**3))
    write(root/'receipt.json',receipt)
    def budget(kind):
        assert sum(receipt['counts'].values())<300 and time.monotonic()-start<21600
        assert sum(p.stat().st_size for p in root.rglob('*') if p.is_file())<4*1024**3
        receipt['counts'][kind]+=1
    def timeout(*args):
        raise TimeoutError('six allocated GPU-hour ceiling')
    signal.signal(signal.SIGALRM,timeout); signal.alarm(21600)
    saved_gate={}
    try:
        for condition in ['tied-shared','tied-split','untied']:
            model_key='untied' if condition=='untied' else 'tied'
            qwen,identity=load_model(model_key,'cuda:0'); model=qwen.model
            emb,head=model.get_input_embeddings(),model.get_output_embeddings()
            if condition=='tied-split':
                head.shared_embed_delta=torch.nn.Parameter(head.shared_embed_delta.detach().clone())
            params={'shared':emb.shared_embed_delta} if condition=='tied-shared' else {'in':emb.shared_embed_delta,'out':head.shared_embed_delta}
            for p in params.values(): p.requires_grad_(True)
            assert {id(p) for p in model.parameters() if p.requires_grad}=={id(p) for p in params.values()}
            original={p:thash(v) for p,v in params.items()}
            ids=head.selected_token_ids.tolist(); coords=[qwen.tokenizer.convert_tokens_to_ids(f'<|coord_{i}|>') for i in range(1000)]
            coordinate_rows=[ids.index(i) for i in coords]
            weights={'in':emb(torch.tensor(ids,device='cuda:0')).detach().cpu(),'out':(head.base.weight[ids]+head.shared_embed_delta).detach().cpu()}
            if condition=='tied-shared':
                assert torch.equal(weights['in'],weights['out']);weights={'shared':weights['in']}
            dest=root/condition;dest.mkdir()
            torch.save(dict(weights=weights,coordinate_rows=coordinate_rows,selected_ids=ids),dest/'weights.pt')
            write(dest/'identity.json',identity)
            def terms(route):
                requests,_=build_bound_native_requests(qwen,panel['configs'][model_key],[route['case']])
                batch=prepare_native_inputs(qwen.processor,requests,device='cuda:0',record_media_identity=True)
                assert list(batch.prompt_token_ids[0])==route['prompt_token_ids']
                assert batch.media_sha256[0]==route['image_identity']['executed_media_sha256']
                assert list(batch.image_grids[0])==route['image_identity']['observed_image_grid_thw']
                replay=prepare_replay(model,batch.inputs,prompt_token_ids=route['prompt_token_ids'],continuation_token_ids=route['continuation_token_ids'])
                budget('forwards'); logits=replay.aligned_logits(model(**replay.inputs).logits).float()
                ce,counts=masked_ce_loss(logits,replay.target_ids,route['ce_weights'])
                axis=raw_axis_validity_hinge(logits,route['trusted_boxes'],coordinate_token_ids=coords,coordinate_bin_values=list(range(1000)),margin=1/999)
                # A zero-box segment keeps its zero contribution in the global denominator.
                if not axis.requires_grad: axis=logits.sum()*0
                return {'ce':ce,'axis':axis},counts,logits
            def gradients(losses):
                answer={}
                for objective in ['ce','axis']:
                    budget('backwards')
                    values=torch.autograd.grad(losses[objective],tuple(params.values()),retain_graph=objective=='ce',allow_unused=False)
                    assert all(torch.isfinite(g).all() for g in values)
                    answer[objective]={p:g.detach().float().cpu() for p,g in zip(params,values)}
                return answer
            # First fixed teacher is the supervised gate slice; same slice in all conditions.
            losses,_,logits=terms(routes[0]); gate_logits=logits.detach().cpu(); gate_grad=gradients(losses)
            saved_gate[condition]=dict(logits=gate_logits,gradients=gate_grad)
            torch.save(dict(gradients=gate_grad,logits_sha256=thash(gate_logits),selected_logits=gate_logits[:,ids]),dest/'gate.pt')
            if condition=='tied-split':
                forward=residual(saved_gate['tied-shared']['logits'],gate_logits)
                assert forward['max_abs']<=TOL['forward_max_abs'],forward
                sums={o:residual(saved_gate['tied-shared']['gradients'][o]['shared'],gate_grad[o]['in']+gate_grad[o]['out']) for o in gate_grad}
                assert all(r['max_abs']<=TOL['gradient_max_abs'] and r['relative_l2']<=TOL['gradient_relative_l2'] for r in sums.values()),sums
                receipt['gates']['tied_split']=dict(forward=forward,backward=sums)
            del logits,losses,gate_logits
            fd=[]
            for path,param in params.items():
                original_value=param.detach().clone()
                for label,row in [('endpoint',coordinate_rows[0]),('interior',coordinate_rows[500]),('all_selected',None)]:
                    direction=torch.zeros_like(param)
                    pattern=torch.where(torch.arange(param.shape[1],device=param.device)%2==0,1.,-1.)
                    if row is None: direction[:]=pattern
                    else: direction[row]=pattern
                    direction/=direction.norm()
                    values=[]
                    try:
                        for sign in [1,-1]:
                            with torch.no_grad():
                                param.copy_(original_value+sign*TOL['fd_epsilon']*direction)
                                perturbed,_,logits=terms(routes[0]);values.append({o:float(v) for o,v in perturbed.items()})
                                del perturbed,logits
                    finally:
                        with torch.no_grad():param.copy_(original_value)
                    assert thash(param)==original[path]
                    for objective in ['ce','axis']:
                        observed=(values[0][objective]-values[1][objective])/(2*TOL['fd_epsilon'])
                        expected=float((gate_grad[objective][path]*direction.cpu()).sum())
                        bound=TOL['fd_atol']+TOL['fd_rtol']*max(abs(observed),abs(expected))
                        entry=dict(path=path,direction=label,objective=objective,finite_difference=observed,autograd=expected,absolute_error=abs(observed-expected),bound=bound,restored_sha256=thash(param),direction_sha256=thash(direction),plus=values[0][objective],minus=values[1][objective])
                        fd.append(entry);write(dest/'finite-difference.json',fd)
                        assert abs(observed-expected)<=bound,entry
                assert any(e['path']==path and e['objective']=='ce' and abs(e['autograd'])>1e-8 for e in fd),'no CE path sensitivity'
            receipt['gates'][condition]=dict(finite_difference='passed',checks=len(fd))
            write(root/'receipt.json',receipt)
            for index,route in enumerate(routes):
                losses,counts,logits=terms(route);values={o:float(v.detach()) for o,v in losses.items()};raw=gradients(losses)
                torch.save(dict(image_id=route['image_id'],losses=values,gradients=raw,complete_boxes=len(route['trusted_boxes']),incomplete_boxes=0,
                    active_tokens=counts['active_tokens'],target_sha256=hashlib.sha256(json.dumps(route['continuation_token_ids']).encode()).hexdigest(),
                    mask_sha256=hashlib.sha256(json.dumps(route['ce_weights']).encode()).hexdigest()),dest/f'scene-{index:02d}.pt')
                del losses,logits,raw
                assert all(thash(v)==original[p] for p,v in params.items())
                write(root/'receipt.json',receipt)
            receipt['complete_passes']+=1
            del terms,gradients,params,emb,head,model,qwen,param,original_value,direction
            gc.collect();torch.cuda.empty_cache()
        reduce(root);receipt['status']='candidate_completed'
    except BaseException as error:
        receipt['status']='technical_invalid';receipt['error']=repr(error)
        raise
    finally:
        signal.alarm(0)
        receipt['allocated_gpu_seconds']=time.monotonic()-start
        receipt['peak_cuda_bytes']=torch.cuda.max_memory_allocated()
        receipt['retained_bytes']=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
        receipt['job_closed']=True;write(root/'receipt.json',receipt)


def self_check():
    import tempfile
    g=torch.tensor([[1.,0.],[0.,2.]]); assert residual(g,g)=={'max_abs':0.,'relative_l2':0.}
    p=torch.nn.Parameter(torch.tensor([2.,3.]));q=torch.nn.Parameter(p.detach().clone())
    shared=torch.autograd.grad((p*p).sum(),p)[0]
    a,b=torch.autograd.grad((p*q).sum(),(p,q));assert torch.equal(shared,a+b)
    assert not torch.equal(shared,a), 'sum gate must reject missing output path'
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp);w=torch.ones(1004,2)
        for condition in ['tied-shared','tied-split','untied']:
            dest=root/condition;dest.mkdir();paths=['shared'] if condition=='tied-shared' else ['in','out']
            torch.save(dict(weights={p:w for p in paths},coordinate_rows=list(range(1000))),dest/'weights.pt')
            for i in range(18):
                gradients={'ce':{p:w*(i+1)*(2 if p=='shared' else 1) for p in paths},'axis':{p:torch.zeros_like(w) for p in paths}}
                torch.save(dict(image_id=i,gradients=gradients,complete_boxes=31 if i<17 else 43,incomplete_boxes=0,active_tokens=i+1,
                    losses=dict(ce=float(i+1),axis=0.),target_sha256=str(i),mask_sha256=str(i)),dest/f'scene-{i:02d}.pt')
        summary=reduce(root)
        assert summary['conditions']['untied']['losses']['ce']==9.5
        assert summary['conditions']['untied']['tables']['ce']['in_out']['cosine']>.99999
        assert summary['conditions']['untied']['tables']['axis']['in_out']['cosine'] is None
        aggregate=torch.load(root/'aggregate.pt',weights_only=True)
        assert torch.equal(aggregate['tied-shared']['ce']['shared'],w*19)
    print('CPU gradient identity, equal-scene reducer, and zero-norm checks passed')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','run','reduce','self-check'])
    parser.add_argument('--root',type=Path,default=ROOT);parser.add_argument('--panel',type=Path);parser.add_argument('--routes',type=Path);parser.add_argument('--gate',type=Path)
    args=parser.parse_args()
    if args.command=='self-check':self_check()
    elif args.command=='prepare':prepare(args.panel,args.routes)
    elif args.command=='reduce':reduce(args.root)
    else:execute(args.panel,args.routes,args.gate,args.root)
