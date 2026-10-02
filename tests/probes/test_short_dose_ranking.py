"""CPU doubles exercise the released caller and final decision consumer."""
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import signal
import time
from types import SimpleNamespace
from uuid import UUID

import pytest
import torch

from probes import short_dose_ranking as x


def write(path, value):
    Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


@pytest.fixture(scope='module')
def prepared(tmp_path_factory):
    output=tmp_path_factory.mktemp('short-dose-prep')/'preparation'
    c=x.prepare(output)
    q,inputs,requests,reports=x.frontend(c)
    return c,q,inputs,requests,reports


def test_four_phase_callers_consumer_and_resigned_false_results(prepared,tmp_path,monkeypatch):
    from src.qwen import vllm_rollout
    c,q,inputs,requests,reports=prepared
    c=copy.deepcopy(c); c.update(source={'commit':'CPU-double'},native_released=True,
        execution_checkout=str(x.ROOT),output_root=str(tmp_path))
    output=tmp_path/'package'; output.mkdir(); path=tmp_path/'manifest.json'; write(path,c)
    expected_sha=x.b.sha(path)
    calls=[]; resets=[]; engines=[]; forwarded=[]; optimizer_records=[]; models=[]; modes=[]
    monkeypatch.setattr(x,'validate',lambda *a,**k:c)
    monkeypatch.setattr(x,'frontend',lambda _:(q,inputs,requests,reports))
    monkeypatch.setattr(x,'resources',lambda *a,**k:dict(seconds=.1,peak_rss_kib=1,cuda_peak_allocated=None,cuda_peak_reserved=None))
    monkeypatch.setattr(torch.cuda,'manual_seed_all',lambda _:None)
    monkeypatch.setattr(torch.cuda,'empty_cache',lambda:None)
    monkeypatch.setattr(x.b.online,'set_checkpointing',lambda *a:None)
    monkeypatch.setattr(x.b.online,'native_batch',lambda q,item:SimpleNamespace(inputs=dict(
        input_ids=torch.tensor([item['prompt_token_ids']]),image_grid_thw=torch.tensor([item['image_grid_thw']]),
        pixel_values=torch.ones(1),position_ids='stale',past_key_values='stale',cache_position='stale')))
    threads=torch.get_num_threads(); torch.set_num_threads(1)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_weight=torch.nn.Parameter(torch.tensor([.1]))
            self.input_delta=torch.nn.Parameter(torch.tensor([.2]))
            self.output_delta=torch.nn.Parameter(torch.tensor([.3]))
            self.register_buffer('mask',torch.zeros(c['vocab_size']))
            self.mask[c['coordinate_ids'][:999]]=1
        def get_rope_index(self,ids,mm_token_type_ids,**kwargs):
            return torch.arange(ids.shape[1]).view(1,1,-1).expand(3,1,-1),None
        def forward(self,input_ids,position_ids,logits_to_keep,use_cache,**kwargs):
            assert logits_to_keep==1 and use_cache is False
            assert 'past_key_values' not in kwargs and 'cache_position' not in kwargs
            assert torch.equal(position_ids[0,0],torch.arange(input_ids.shape[1]))
            context=next(z for z in c['contexts'] if z['processed_prompt_token_ids']==input_ids[0].tolist())
            forwarded.append(context['context_id']); modes.append((self.training,torch.is_grad_enabled()))
            # Actual last-logit differentiable replay with a history-sensitive CPU double.
            adjustment=(self.lora_weight+self.input_delta+self.output_delta)*(1+context['offset']*.1)
            return SimpleNamespace(logits=(self.mask*adjustment).view(1,1,-1))
    def compose(checkpoint,evaluation):
        assert str(checkpoint)==c['checkpoint'] and evaluation is False
        assert not engines or engines[-1].closed
        model=Model(); models.append(model); resets.append([float(t.detach()) for t in model.parameters()])
        delta=SimpleNamespace(delta_tensors=lambda:dict(input=model.input_delta,output=model.output_delta))
        return SimpleNamespace(model=model,tokenizer=q.tokenizer),delta,{'CPU_model_double':True}
    def binding(q,delta,_):
        values=list(delta.delta_tensors().values())
        return list(q.model.parameters()),[q.model.lora_weight],values
    def save(q,delta,directory):
        directory.mkdir(parents=True,exist_ok=False)
        data=[float(t.detach()) for t in q.model.parameters()]
        write(directory/'payload.json',data)
        write(directory/'identity.json',{'payload.json':x.b.sha(directory/'payload.json')})
    def verify_start(checkpoint,exported):
        assert str(checkpoint)==c['checkpoint']
        assert x.b.load(exported/'payload.json')==pytest.approx([.1,.2,.3])
    original_optimizer=torch.optim.AdamW
    def optimizer(*a,**k):
        result=original_optimizer(*a,**k)
        assert not result.state
        optimizer_records.append(result)
        return result
    monkeypatch.setattr(x.p,'compose',compose)
    monkeypatch.setattr(x.p,'save_checkpoint',save)
    monkeypatch.setattr(x,'live_training_binding',binding)
    monkeypatch.setattr(x.b.online,'verify_start_export',verify_start)
    monkeypatch.setattr(torch.optim,'AdamW',optimizer)

    class Engine:
        def __init__(self,**kwargs):
            assert len(forwarded)==54 and x.b.load(output/'train/complete.json')['HF_closed_before_publish'] is True
            assert kwargs['max_num_seqs']==1 and kwargs['max_model_len']==4456
            assert kwargs['kv_cache_memory_bytes']==2*1024**3 and kwargs['timeout']==1800
            assert not engines or engines[-1].closed
            self.identity=kwargs['identity']; self.receipts=[]; self.closed=False; self.score_index=0; self.natural_index=0
            self._process=SimpleNamespace(is_alive=lambda:not self.closed)
            self.arm=(['anchor']+x.ENDPOINTS)[len(engines)]
            physical=dict(uuid='GPU-'+str(UUID(int=1)),pci_domain_id=0,pci_bus_id=1,pci_device_id=0)
            request=dict(schema='coordexp-vllm-device-1',rank=0,device=0,physical_token='0',
                parent=dict(pid=1,ppid=99,nspid='NSpid: 1',visibility=None,physical=physical))
            self.startup=dict(identity=self.identity,device=dict(requested=request,child=dict(pid=2,ppid=1,nspid='NSpid: 2'),
                inherited_visibility='0',effective_visibility='0',logical_device=0,physical=physical))
            engines.append(self)
        def __enter__(self): return self
        def __exit__(self,*_): self.closed=True
        def receipt(self,operation):
            self.receipts.append(dict(operation=operation,identity=self.identity,coordinate_output_norm=dict(
                mode='off',identity=self.identity,coordinate_ids=c['coordinate_ids'],calls=1,
                first_call=dict(scaling_active=False,non_coordinate_unchanged=True))))
        def configure_coordinate_output_norm(self,mode,ids,**kw):
            assert mode=='off' and ids==c['coordinate_ids']; self.receipt('coordinate_output_norm')
        def generate_exact(self,request,**kw):
            context=c['contexts'][self.score_index]
            assert len(request)==1 and kw['budgets']==[1] and kw['full_scores'] is True
            assert kw['extensions']==[context['extension']] and kw['identity']==self.identity
            assert kw['chat_token_ids']==[next(r['unexpanded_chat_token_ids'] for r in reports if r['image_id']==context['image_id'])]
            assert self.natural_index==0
            # Baseline mixes legal and illegal. Held-out retention can be lost.
            legal=(self.score_index%2==1) if self.arm=='anchor' else (self.score_index!=4 if self.arm=='step1' else True)
            token=c['coordinate_ids'][0 if legal else 999]
            scores={i:-math.log(c['vocab_size']) for i in range(c['vocab_size'])}
            calls.append((self.arm,'score',context['context_id'])); self.score_index+=1; self.receipt('generate_exact')
            return [dict(request_id=request[0].request_id,token_ids=[token],stop_reason='length',full_scores=scores,
                processed_prompt_token_ids=context['processed_prompt_token_ids'],vocab_size=c['vocab_size'],
                score_semantics='native_raw_logprobs_full_vocabulary')]
        def generate(self,request,**kw):
            assert len(request)==1 and self.score_index==10 and kw['budgets']==[3084]
            image=c['images'][self.natural_index]; assert request[0].request_id==inputs[image]['request_id']
            tokens=[c['eos_token_id']]
            if image in [7511,351017] and self.arm!='anchor':
                raw=x.b.load(c['raw_paths'][str(image)])
                tokens=raw['token_ids'][:next(z['earlier_row']['positions'][-1]+1 for z in c['contexts'] if z['image_id']==image)]+tokens
            calls.append((self.arm,'natural',image)); self.natural_index+=1; self.receipt('generate')
            return [SimpleNamespace(request_id=request[0].request_id,token_ids=tokens,stop_reason='im_end')]
    monkeypatch.setattr(vllm_rollout,'VllmDoraRollout',Engine)
    try:
        x.train(path,output,expected_sha)
        for endpoint in ['anchor']+x.ENDPOINTS: x.native(path,output,expected_sha,endpoint)
        result=x.readback(path,output,expected_sha)
    finally:
        torch.set_num_threads(threads)
    assert len(resets)==len(optimizer_records)==1
    assert [g['lr'] for g in optimizer_records[0].param_groups]==[1e-5,5e-6]
    assert [int(v['step']) for v in optimizer_records[0].state.values()]==[4]*3
    assert len(forwarded)==54 and len(calls)==84 and all(e.closed for e in engines)
    contexts=[z['context_id'] for z in c['contexts']]
    schedule=[z['context_id'] for z in x.r.views(c,'R-single')]
    assert forwarded==contexts+schedule+contexts+schedule*3+contexts
    assert modes==[(False,False)]*10+[(True,True)]*6+[(False,False)]*10+[(True,True)]*18+[(False,False)]*10
    assert result['counters']==dict(optimizer_steps=4,training_replays=24,HF_diagnostics=30,
        score_requests=30,continuation_requests=54,requests=84,generated_tokens=result['counters']['generated_tokens'])
    assert result['conditional']['step1']['7511-626/original']['baseline_illegal_denominator']==1
    assert result['conditional_natural_recovery_credit'] is False
    assert set(result['natural_transitions'])=={'step1','step4','step1-to-step4'}
    # Independent uninterrupted AdamW reference: checkpoint/HF publication must leave update2 unchanged.
    reference=Model()
    ref_optimizer=original_optimizer([dict(params=[reference.lora_weight],lr=1e-5),
        dict(params=[reference.input_delta,reference.output_delta],lr=5e-6)],betas=(.9,.999),eps=1e-8,weight_decay=0)
    for step in range(4):
        ref_optimizer.zero_grad(set_to_none=True)
        for _ in range(6):
            (torch.nn.functional.softplus(1-(reference.lora_weight+reference.input_delta+reference.output_delta))/6).backward()
        torch.nn.utils.clip_grad_norm_(list(reference.parameters()),1,error_if_nonfinite=True)
        ref_optimizer.step()
    assert all(torch.equal(a,z) for a,z in zip(reference.parameters(),models[0].parameters(),strict=True))
    trace=[x.b.load(output/'train'/f'update-{i:02d}.json') for i in range(1,5)]
    assert all(trace[i]['before']==trace[i-1]['after'] for i in range(1,4))
    publication=x.b.load(output/'train/publication-1.json')
    assert publication['before']==publication['after']==trace[0]['after']==trace[1]['before']
    assert x.BOUNDS['generated_tokens']==30+54*3084 and x.BOUNDS['active_seconds']==7200
    assert [command[3] for command in x.phase_commands('manifest','output')]==['train','native','native','native']
    original_complete=x.b.load(output/'complete.json')
    bad=copy.deepcopy(original_complete); bad['conditional']['step1']['7511-626/original']['baseline_illegal_denominator']=99
    write(output/'complete.json',bad)
    with pytest.raises(AssertionError,match='re-signed false terminal'): x.readback(path,output,expected_sha)
    write(output/'complete.json',original_complete)

    def resign(directory,name,value):
        write(directory/name,value)
        receipt=x.b.load(directory/'complete.json'); receipt['artifacts'][name]=x.b.sha(directory/name)
        write(directory/'complete.json',receipt)
    def consume():
        if directory.name=='train': return x.consume_train(c,output,expected_sha)
        return x.consume_native(c,output,'step1',expected_sha,q,inputs)
    directory=output/'native-step1'; name='score-00.json'; original=x.b.load(directory/name)
    for change in ['literal_legal','natural_recovery_credit']:
        bad=copy.deepcopy(original); bad['measurement'][change]=not bad['measurement'][change]; resign(directory,name,bad)
        with pytest.raises(AssertionError,match='false conditional'): consume()
    resign(directory,name,original)
    name='natural-7511.json'; original=x.b.load(directory/name)
    bad=copy.deepcopy(original); bad['measurement']['ids']['category']['retained'].append(12345); resign(directory,name,bad)
    with pytest.raises(AssertionError,match='false natural credit'): consume()
    resign(directory,name,original)
    receipt=x.b.load(directory/'complete.json')
    for field,value in [('arm','step4'),('checkpoint',str(output/'train/checkpoint-4')),('weight_identity','wrong'),('counters',{}),('source',{'commit':'false'})]:
        bad=copy.deepcopy(receipt); bad[field]=value; write(directory/'complete.json',bad)
        with pytest.raises(AssertionError): consume()
    write(directory/'complete.json',receipt)
    directory=output/'train'; name='update-01.json'; original=x.b.load(directory/name)
    for change in ['context_id','weight','loss']:
        bad=copy.deepcopy(original); bad['terms'][0][change]='swapped' if change=='context_id' else 99; resign(directory,name,bad)
        with pytest.raises((AssertionError,TypeError)): consume()
    resign(directory,name,original)
    receipt=x.b.load(directory/'complete.json'); bad=copy.deepcopy(receipt); bad['fresh_optimizer']=False
    write(directory/'complete.json',bad)
    with pytest.raises(AssertionError): consume()
    write(directory/'complete.json',receipt)
    consume()
    # Exchange the actual exported payloads; even the CPU endpoint reader must reject the swap.
    first=directory/'checkpoint-1/payload.json'; fourth=directory/'checkpoint-4/payload.json'
    first_bytes,fourth_bytes=first.read_bytes(),fourth.read_bytes()
    assert first_bytes!=fourth_bytes
    first.write_bytes(fourth_bytes); fourth.write_bytes(first_bytes)
    try:
        with pytest.raises(AssertionError,match='endpoint checkpoint drift'): x.weight_binding(c,output,'step1')
    finally:
        first.write_bytes(first_bytes); fourth.write_bytes(fourth_bytes)
    for name,mutate in [
        ('update-02.json',lambda v:v['before'].update(optimizer_object=-1)),
        ('update-02.json',lambda v:v['before']['optimizer_state']['lora_weight'].update(step=0)),
        ('publication-1.json',lambda v:v.update(training_mode_restored=False)),
        ('publication-1.json',lambda v:v['after']['optimizer_state']['lora_weight']['values'].update(exp_avg='reset'))]:
        original=x.b.load(directory/name); bad=copy.deepcopy(original); mutate(bad); resign(directory,name,bad)
        with pytest.raises(AssertionError): consume()
        resign(directory,name,original)
    receipt=x.b.load(directory/'complete.json')
    for mutation in ['swap','false_counters','false_source']:
        bad=copy.deepcopy(receipt)
        if mutation=='swap': bad['checkpoints']['1'],bad['checkpoints']['4']=bad['checkpoints']['4'],bad['checkpoints']['1']
        elif mutation=='false_counters': bad['counters']['optimizer_steps']=16
        else: bad['source']={'commit':'false'}
        write(directory/'complete.json',bad)
        with pytest.raises(AssertionError): consume()
        write(directory/'complete.json',receipt)
    consume()
    # A real destructive reset in the save hook must fail the actual training caller before update2.
    original_save=x.p.save_checkpoint
    def reset_during_save(q,delta,directory):
        original_save(q,delta,directory)
        if directory.name=='checkpoint-1': optimizer_records[-1].state.clear()
    monkeypatch.setattr(x.p,'save_checkpoint',reset_during_save)
    monkeypatch.setattr(x,'entry',lambda *a:c)
    with pytest.raises(AssertionError,match='checkpoint/diagnostic optimizer continuity'):
        x.train(path,tmp_path/'reset-package',expected_sha)
    assert len(optimizer_records)==2 and not (tmp_path/'reset-package/train/update-02.json').exists()
    write(tmp_path/'cpu-evidence.json',dict(status='CPU-candidate',real_model_loaded=False,
        real_native_launched=False,composition_count=1,optimizer_count=1,
        counters=result['counters'],parameter_equal_to_uninterrupted_reference=True,
        actual_reset_rejected_before_update2=True,actual_checkpoint_swap_rejected=True,
        resigned_false_continuity_mode_checkpoint_counters_source_credit_rejected=True,
        scientific_acceptance=False))


def test_inherited_contract_and_release_fail_closed(prepared,tmp_path):
    c=prepared[0]; path=tmp_path/'contract.json'
    write(path,c)
    x.validate(path,current_source=False)
    with pytest.raises(AssertionError,match='release'): x.validate(path,require_release=True,current_source=False)
    for key,value in [('bounds',dict(c['bounds'],optimizer_steps=16)),('phases',list(reversed(c['phases']))),
        ('contexts',list(reversed(c['contexts']))),('gt_role','training'),('training',{})]:
        bad=copy.deepcopy(c); bad[key]=value; write(path,bad)
        with pytest.raises(AssertionError): x.validate(path,current_source=False)
    from src.artifacts.git_identity import SourceIdentityError
    write(path,dict(c,source={'schema':'obsolete'}))
    with pytest.raises(SourceIdentityError): x.validate(path)


def test_package_order_and_existing_cleanup_helpers(tmp_path,monkeypatch):
    assert x.cleanup_phase_group is x.r.cleanup_phase_group and x.live_phase_group is x.r.live_phase_group
    waits=[]; commands=[]; consumed=[]
    monkeypatch.setattr(x,'entry',lambda *a:dict(bounds=x.BOUNDS))
    def launch(command,**kwargs):
        commands.append(command)
        assert kwargs['start_new_session'] is True
        return SimpleNamespace(pid=123456,wait=lambda timeout:waits.append(timeout) or 0)
    monkeypatch.setattr(x.subprocess,'Popen',launch)
    monkeypatch.setattr(x,'live_phase_group',lambda _:[])
    monkeypatch.setattr(x,'readback',lambda *a:consumed.append(a))
    x.package(tmp_path/'contract',tmp_path/'package','CPU')
    assert waits==[1800]*4 and len(consumed)==1
    assert [c[3] for c in commands]==['train','native','native','native']
    assert [c[-1] for c in commands[1:]]==['anchor','step1','step4']
    assert all('CPU' in c for c in commands)
