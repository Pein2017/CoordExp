"""Frozen global-L2 controls on the accepted417044 coordinate-cache seam."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
import traceback

import torch

from .instance_state import (ROOT as PARENT, EOS, cache_slices, digest, file_hash,
    generation_inputs, grounding_mask, require, transplant, write)

ROOT = PARENT/'amplitude-control'
ARMS = ('native_self','original_a','b_to_a','background_to_a','a_to_b','random_to_a','a_k_only','a_v_only')
SEED = 20260912


def flatten_slices(slices):
    return torch.cat([tensor.detach().cpu().reshape(-1).to(torch.float64)
                      for pair in slices for tensor in pair])


def norm64(vector):
    vector = vector.to(torch.float64)
    require(bool(torch.isfinite(vector).all()), 'nonfinite cache vector')
    return float(torch.linalg.vector_norm(vector))


def scaled_delta(vector, target_norm):
    source_norm = norm64(vector)
    require(source_norm > 0 and target_norm > 0, 'zero-norm scaling is unsupported')
    return vector.to(torch.float64)*(target_norm/source_norm)


def diagnostics(vector, template):
    vector = vector.to(torch.float64)
    require(vector.numel()==sum(t.numel() for pair in template for t in pair),'diagnostic dimensions mismatch')
    result = {'l2':norm64(vector),'rms':float(vector.square().mean().sqrt()),
              'max_abs':float(vector.abs().max()),'scalars':vector.numel(),'per_layer':[]}
    at = 0
    for index,pair in enumerate(template):
        record = {'layer':index}
        for kind,tensor in zip(('k','v'),pair):
            part = vector[at:at+tensor.numel()]
            record[kind] = {'l2':norm64(part),'rms':float(part.square().mean().sqrt()),'max_abs':float(part.abs().max())}
            at += tensor.numel()
        result['per_layer'].append(record)
    return result


def slices_from_vector(vector, template):
    result, at = [], 0
    for pair in template:
        values = []
        for original in pair:
            values.append(vector[at:at+original.numel()].reshape(original.shape).to(original.dtype))
            at += original.numel()
        result.append(tuple(values))
    require(at==vector.numel(),'slice reconstruction size mismatch')
    return result


def branch_slices(captured, arm):
    require(arm in ARMS,'unregistered amplitude arm')
    native = captured['native']
    base = flatten_slices(native)
    da = flatten_slices(captured['owner_a'])-base
    db = flatten_slices(captured['owner_b'])-base
    dg = flatten_slices(captured['background'])-base
    na, nb, ng = norm64(da), norm64(db), norm64(dg)
    require(min(na,nb,ng)>0,'zero-norm donor blocks frozen contrast')
    scale = None
    if arm=='native_self':
        result = native
    elif arm=='original_a':
        result = captured['owner_a']  # exact donor bytes, not subtract/add round-trip
    elif arm=='a_k_only':
        result = [(a[0],n[1]) for a,n in zip(captured['owner_a'],native)]
    elif arm=='a_v_only':
        result = [(n[0],a[1]) for a,n in zip(captured['owner_a'],native)]
    else:
        if arm=='b_to_a':
            delta, target, scale = db,na,na/nb
        elif arm=='background_to_a':
            delta, target, scale = dg,na,na/ng
        elif arm=='a_to_b':
            delta,target,scale = da,nb,nb/na
        else:
            generator = torch.Generator(device='cpu').manual_seed(SEED)
            delta = torch.randn(base.shape,dtype=torch.float64,generator=generator)
            target = na
            scale = na/norm64(delta)
        result = slices_from_vector(base+scaled_delta(delta,target),native)
        observed = norm64(flatten_slices(result)-base)
        require(abs(observed-target)/target < 5e-5,'float32 realized delta norm departed from frozen target')
    delta = flatten_slices(result)-base
    return result, {'arm':arm,'scale':scale,'norm_rule':'single global float64 L2 over all layers,K,V,4coordinate positions',
                    'a_norm':na,'b_norm':nb,'background_norm':ng,'random_seed':SEED if arm=='random_to_a' else None,
                    'realized_delta':diagnostics(delta,native)}


def prepare():
    old = json.loads((PARENT/'panel.json').read_text())
    case = next(c for c in old['cases'] if c['case_id']=='417044')
    require(len(case['carriers']['coordinates'])==4,'coordinate carrier changed')
    refs = {'native_self':PARENT/'panel-v1/417044-self-coordinates.json',
            'original_a':PARENT/'panel-v1/417044-owner_a-coordinates.json'}
    references = {k:{'path':str(p),'sha256':file_hash(p),'suffix_ids':json.loads(p.read_text())['suffix_ids']} for k,p in refs.items()}
    packet = {'schema':'owner_grounding_global_l2.v1','config':old['config'],'anchor_adapter':old['anchor_adapter'],
              'case':case,'parent_packet':str(PARENT/'panel.json'),'parent_sha256':file_hash(PARENT/'panel.json'),
              'references':references,'arms':list(ARMS),'norm_rule':'global_joint_float64',
              'random_seed':SEED,'budget':3084-len(case['history_ids'])-1,
              'gate':{'gpu':'5','arms':['native_self','original_a']},
              'workers':[{'gpu':gpu,'arms':[arm]} for gpu,arm in zip(
                  ('0','1','4','5','6','7'),
                  ('b_to_a','background_to_a','a_to_b','random_to_a','a_k_only','a_v_only'))],
              'source_files':{str(Path(__file__).resolve()):file_hash(__file__),
                 str(Path(__file__).with_name('instance_state.py').resolve()):file_hash(Path(__file__).with_name('instance_state.py'))}}
    write(ROOT/'packet.json',packet)
    return {'packet':str(ROOT/'packet.json'),'sha256':file_hash(ROOT/'packet.json'),'arms':list(ARMS),
            'max_decode_tokens':packet['budget']*8,'prefills':11,'model_loads':8,'assigned_gpus':[0,1,4,5,6,7]}


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def load_case(packet, out, receipt):
    from src.config.inference import InferConfig
    from probes.dora_owner_learning.runtime import load_policy
    from probes.source_rweak_row_cross.run import build_requests
    from src.qwen.native import prepare_native_inputs
    require(torch.cuda.device_count()==1,'exactly one visible GPU required per worker')
    qwen, identity = load_policy(InferConfig.model_validate(packet['config']),device=torch.device('cuda:0'))
    require(identity['model_identity']['adapter']['adapter_path']==packet['anchor_adapter'],'anchor changed')
    write(out/'model.json',identity)
    receipt['model_forwards'],receipt['vision_forwards'] = 0,0
    def counter(*_): receipt['model_forwards'] += 1
    def vision_counter(*_): receipt['vision_forwards'] += 1
    qwen.model.register_forward_pre_hook(counter)
    vision = [m for m in qwen.model.modules() if type(m).__name__=='Qwen3VLVisionModel']
    require(len(vision)==1,'vision module mismatch')
    vision[0].register_forward_pre_hook(vision_counter)
    case = packet['case']
    requests,_ = build_requests(qwen,packet['config'],[case['source_case']])
    batch = prepare_native_inputs(qwen.processor,requests,device=torch.device('cuda:0'),record_media_identity=True)
    require(list(batch.prompt_token_ids[0])==case['prompt_ids'],'prompt changed')
    ids = torch.tensor([case['prompt_ids']+case['history_ids']],device='cuda:0')
    return qwen,batch,ids


def do_prefill(qwen,batch,ids):
    observed = []
    modules = [m for m in qwen.model.modules() if type(m).__name__=='Qwen3VLTextModel']
    require(len(modules)==1,'text model missing')
    def record_positions(_m,_a,kwargs):
        p = kwargs['position_ids']
        observed.append({'shape':list(p.shape),'sha256':tensor_hash(p)})
    hook = modules[0].register_forward_pre_hook(record_positions,with_kwargs=True)
    try:
        result = qwen.model(**generation_inputs(batch,ids),use_cache=True,return_dict=True,logits_to_keep=1)
    finally:
        hook.remove()
    require(len(observed)==1,'prefill position capture mismatch')
    return result,observed[0]


def stage(packet_path,out_dir,kind,arms=(),capture_dir=None):
    packet = json.loads(Path(packet_path).read_text())
    require(file_hash(packet['parent_packet'])==packet['parent_sha256'],'first panel changed')
    for ref in packet['references'].values():
        require(file_hash(ref['path'])==ref['sha256'],'reference output changed')
    out = Path(out_dir);out.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(__file__,out/'runner.py');shutil.copyfile(packet_path,out/'packet.json')
    shutil.copyfile(Path(__file__).with_name('instance_state.py'),out/'instance_state_dependency.py')
    receipt = {'status':'running','kind':kind,'arms':list(arms),'packet_sha256':file_hash(packet_path),
               'runner_sha256':file_hash(__file__),'gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'cells':[]}
    start = time.monotonic();write(out/'receipt.json',receipt)
    try:
        qwen,batch,ids = load_case(packet,out,receipt)
        case = packet['case'];positions=case['carriers']['coordinates']
        with torch.inference_mode():
            prefill,position_record = do_prefill(qwen,batch,ids)
            baseline = prefill.past_key_values
            rope = qwen.model.model.rope_deltas.clone();del prefill
            receipt['mrope_positions'] = position_record
            receipt['rope_delta_sha256'] = tensor_hash(rope)
            if kind=='capture':
                captured = {'native':[(k.cpu(),v.cpu()) for k,v in cache_slices(baseline,positions)]}
                receipt['masks'] = {}
                for arm,keys in case['regions'].items():
                    receipt['masks'][arm] = {}
                    with grounding_mask(qwen.model,case['queries'],keys,ids.shape[1],receipt['masks'][arm]):
                        donor,donor_positions = do_prefill(qwen,batch,ids)
                    require(donor_positions==position_record,'donor MRoPE positions changed')
                    require(torch.equal(qwen.model.model.rope_deltas,rope),'donor delta changed')
                    captured[arm] = [(k.cpu(),v.cpu()) for k,v in cache_slices(donor.past_key_values,positions)]
                    del donor
                payload = {f'{source}.layer{layer:02d}.{kind}':tensor.contiguous()
                           for source,slices in captured.items() for layer,pair in enumerate(slices)
                           for kind,tensor in zip(('k','v'),pair)}
                from safetensors.torch import save_file
                save_file(payload,str(out/'donor_slices.safetensors'))
                receipt['payload_sha256'] = file_hash(out/'donor_slices.safetensors')
                receipt['donor_diagnostics'] = {k:diagnostics(flatten_slices(v)-flatten_slices(captured['native']),captured['native']) for k,v in captured.items()}
                plans = {arm:branch_slices(captured,arm)[1] for arm in ARMS}
                write(out/'frozen_scales.json',plans)
                receipt['scales_sha256'] = file_hash(out/'frozen_scales.json')
            else:
                capture = Path(capture_dir);old=json.loads((capture/'receipt.json').read_text())
                require(old['status']=='complete' and old['packet_sha256']==receipt['packet_sha256'],'capture not admitted')
                require(old['payload_sha256']==file_hash(capture/'donor_slices.safetensors'),'donor payload changed')
                require(old['mrope_positions']==position_record and old['rope_delta_sha256']==tensor_hash(rope),'worker positions differ')
                from safetensors.torch import load_file
                data = load_file(str(capture/'donor_slices.safetensors'))
                captured = {name:[(data[f'{name}.layer{i:02d}.k'],data[f'{name}.layer{i:02d}.v']) for i in range(len(baseline.layers))] for name in ('native','owner_a','owner_b','background')}
                native = cache_slices(baseline,positions)
                require(all(torch.equal(a.cpu(),b) for p,q in zip(native,captured['native']) for a,b in zip(p,q)),'worker native cache bytes differ from capture')
                fullids=torch.cat([ids,torch.tensor([[case['opener']]],device=ids.device)],dim=1)
                frozen = json.loads((capture/'frozen_scales.json').read_text())
                for arm in arms:
                    chosen,stats = branch_slices(captured,arm)
                    require(stats==frozen[arm],'branch scale changed after capture')
                    cache=copy.deepcopy(baseline)
                    changed=transplant(cache,[(k.to(ids.device),v.to(ids.device)) for k,v in chosen],positions)
                    qwen.model.model.rope_deltas=rope.clone()
                    consumed=[]
                    def hook(_m,_a,kw):
                        consumed.append({'length':kw['input_ids'].shape[1],'cache_length':kw['past_key_values'].get_seq_length(),'position':kw['cache_position'].tolist()})
                    handle=qwen.model.register_forward_pre_hook(hook,with_kwargs=True)
                    before=receipt['vision_forwards']
                    try:
                        generated=qwen.model.generate(**generation_inputs(batch,fullids),past_key_values=cache,
                          cache_position=torch.tensor([ids.shape[1]],device=ids.device),max_new_tokens=packet['budget'],
                          do_sample=False,repetition_penalty=1.0,eos_token_id=EOS,pad_token_id=qwen.tokenizer.pad_token_id,
                          return_dict_in_generate=False,output_scores=False,output_logits=False,output_hidden_states=False,output_attentions=False)
                    finally:handle.remove()
                    require(receipt['vision_forwards']==before,'decode repeated vision')
                    require(consumed[0]=={'length':1,'cache_length':ids.shape[1],'position':[ids.shape[1]]},'cached opener mismatch')
                    suffix=generated[0,fullids.shape[1]:].tolist()
                    if arm in packet['references']:
                        require(suffix==packet['references'][arm]['suffix_ids'],arm+' exact full reference reproduction failed')
                    cell={'arm':arm,'suffix_ids':suffix,'prefix_ids':case['history_ids']+[case['opener']],
                          'stop':'eos' if suffix[-1]==EOS else 'length','text':qwen.tokenizer.decode(case['history_ids']+[case['opener']]+suffix,skip_special_tokens=False),
                          'delta_stats':stats,'cache_change':changed,'first_consumption':consumed[0],'decode_forwards':len(consumed)}
                    write(out/(arm+'.json'),cell);receipt['cells'].append({'arm':arm,'sha256':file_hash(out/(arm+'.json'))})
                    write(out/'receipt.json',receipt);del cache,generated
        receipt['status']='complete'
    except BaseException:
        receipt.update(status='technical_invalid',traceback=traceback.format_exc());raise
    finally:
        receipt.update(wall_seconds=time.monotonic()-start,peak_cuda_allocated=torch.cuda.max_memory_allocated(),
                       peak_cuda_reserved=torch.cuda.max_memory_reserved(),peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        write(out/'receipt.json',receipt)


def launch(packet_path,out_dir):
    packet=json.loads(Path(packet_path).read_text());out=Path(out_dir);out.mkdir(parents=True,exist_ok=False)
    write(out/'launch.json',{'status':'started','packet':str(packet_path),'packet_sha256':file_hash(packet_path),'stages':[]})
    def start(name,gpu,kind,arms=()):
        command=[sys.executable,'-m','probes.parallel_owner_research.instance_state_amplitude',kind,'--packet',str(packet_path),'--out-dir',str(out/name)]
        if kind=='worker':command+=['--capture-dir',str(out/'capture'),'--arms',*arms]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu));log=(out/(name+'.log')).open('w')
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)
        return name,process,log,command
    def finish(item):
        name,process,log,command=item;code=process.wait();log.close()
        (out/(name+'.exit')).write_text(str(code)+'\n')
        require(code==0,f'{name} exited {code}; log retained; no retry')
        return {'name':name,'exit':code,'command':command,'receipt_sha256':file_hash(out/name/'receipt.json')}
    completed=[]
    try:
        completed.append(finish(start('capture',5,'capture')))
        completed.append(finish(start('gate',5,'worker',packet['gate']['arms'])))
        workers=[start('worker-'+w['gpu'],w['gpu'],'worker',w['arms']) for w in packet['workers']]
        # All fixed workers are launched before any blocking wait; one owner per GPU.
        errors=[]
        for worker in workers:
            try:completed.append(finish(worker))
            except Exception as exc:errors.append(str(exc))
        require(not errors,'; '.join(errors))
        write(out/'launch.json',{'status':'complete','packet':str(packet_path),'packet_sha256':file_hash(packet_path),'stages':completed})
    except BaseException:
        write(out/'launch.json',{'status':'technical_invalid','packet':str(packet_path),'packet_sha256':file_hash(packet_path),'stages':completed,'traceback':traceback.format_exc()});raise


def consume(packet_path,out_dir):
    from tokenizers import Tokenizer
    from probes.source_rweak_row_cross.run import native_record
    from probes.dora_owner_learning.candidate_opportunity import score
    packet=json.loads(Path(packet_path).read_text());out=Path(out_dir)
    launch=json.loads((out/'launch.json').read_text())
    require(launch['status']=='complete' and launch['packet_sha256']==file_hash(packet_path),'launcher incomplete/packet mismatch')
    case=packet['case'];tok=Tokenizer.from_file(packet['config']['model']['base_model']+'/tokenizer.json')
    records=[]
    for stage_record in launch['stages']:
        directory=out/stage_record['name'];receipt=json.loads((directory/'receipt.json').read_text())
        require(stage_record['receipt_sha256']==file_hash(directory/'receipt.json'),'stage receipt changed')
        require(receipt['status']=='complete','stage incomplete')
        for ref in receipt['cells']:
            path=directory/(ref['arm']+'.json');require(file_hash(path)==ref['sha256'],'branch bytes changed')
            cell=json.loads(path.read_text());action=cell['prefix_ids']+cell['suffix_ids']
            require(cell['prefix_ids']==case['history_ids']+[case['opener']],'branch prefix changed')
            require(tok.decode(action,skip_special_tokens=False)==cell['text'],'token/text mismatch')
            parsed=native_record(cell['text'],case['source_case'],case['golden'],cell['stop'])
            result={'arm':cell['arm'],'parsed':parsed,'score':score(parsed,seed=None,length=len(action),stop=cell['stop']),
                    'raw_row_starts':action.count(case['opener']),'delta_stats':cell['delta_stats'],'cell_path':str(path),
                    'equal_original_a':cell['suffix_ids']==packet['references']['original_a']['suffix_ids'],
                    'equal_native':cell['suffix_ids']==packet['references']['native_self']['suffix_ids']}
            records.append(result)
    require(sorted(r['arm'] for r in records)==sorted(ARMS),'full branch coverage failed')
    native=next(r['score'] for r in records if r['arm']=='native_self')
    for result in records:
        result['gains']={str(t):sorted(set(result['score'][str(t)]['owners'])-set(native[str(t)]['owners'])) for t in (50,60,80)}
        result['losses']={str(t):sorted(set(native[str(t)]['owners'])-set(result['score'][str(t)]['owners'])) for t in (50,60,80)}
    write(out/'consumer.json',{'status':'cold_consumer_verified','launch_sha256':file_hash(out/'launch.json'),'records':records})
    return {'consumer':str(out/'consumer.json'),'records':len(records)}


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','capture','worker','launch','consume'])
    p.add_argument('--packet',default=str(ROOT/'packet.json'));p.add_argument('--out-dir');p.add_argument('--capture-dir');p.add_argument('--arms',nargs='*',default=[])
    a=p.parse_args()
    if a.command=='prepare':print(json.dumps(prepare(),indent=2))
    else:
        require(a.out_dir,'fresh explicit output required')
        if a.command=='launch':launch(a.packet,a.out_dir)
        elif a.command=='consume':print(json.dumps(consume(a.packet,a.out_dir),indent=2))
        else:stage(a.packet,a.out_dir,a.command,a.arms,a.capture_dir)


if __name__=='__main__':main()
