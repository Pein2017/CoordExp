"""CPU doubles exercise the released caller and final decision consumer."""
import copy
import json
import math
from pathlib import Path
import subprocess
from types import SimpleNamespace
from uuid import UUID

import pytest
import torch

from probes import prefix_exposure_ranking as x


def write(path, value):
    Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


@pytest.fixture(scope='module')
def prepared(tmp_path_factory):
    output=tmp_path_factory.mktemp('prefix-prep')/'preparation'
    c=x.prepare(output)
    q,inputs,requests,reports=x.frontend(c)
    return c,q,inputs,requests,reports


def test_literal_contexts_input_bounds_and_no_label_acquisition(prepared,monkeypatch):
    c,q,inputs,requests,reports=prepared
    original=x.b.load
    def guarded(path):
        assert str(path)!=c['label_path'], 'GT leaked into context/acquisition'
        return original(path)
    monkeypatch.setattr(x.b,'load',guarded)
    derived=x.contexts(c,q.tokenizer)
    assert derived==c['contexts'] and len(derived)==10
    assert [z['offset'] for z in derived]==x.OFFSETS*2
    for z in derived:
        raw=original(c['raw_paths'][str(z['image_id'])])
        diffs=[i for i,(a,b) in enumerate(zip(z['extension'],raw['token_ids'][:z['target_position']],strict=True)) if a!=b]
        assert diffs==([z['changed_position']] if z['offset'] else [])
        assert len(z['extension'])==z['target_position']
        assert z['processed_prompt_token_ids']==inputs[z['image_id']]['prompt_token_ids']+z['extension']
        assert z['causal_logits_position']==len(z['processed_prompt_token_ids'])-1
        assert z['support']==c['coordinate_ids'][:999]
    assert len(inputs)==18 and max(r['prompt_tokens'] for r in reports)==1372
    assert [r['image_id'] for r in reports]==c['images']
    assert max(r['prompt_tokens']+3084 for r in reports)==4456
    assert [r['context_id'] for r in x.views(c,'R-single')]==['7511-626/+0']*3+['351017-1507/+0']*3
    assert [r['offset'] for r in x.views(c,'R-multiple')]==[0,-1,1]*2
    bad=copy.deepcopy(c); bad['raw_paths']['7511']=c['raw_paths']['351017']
    with pytest.raises((AssertionError,StopIteration)): x.contexts(bad,q.tokenizer)


def test_standalone_full_complement_and_literal_ties(prepared):
    c=prepared[0]
    z=torch.zeros(c['vocab_size'],requires_grad=True)
    with torch.no_grad():
        z[c['coordinate_ids'][998]]=1
        z[0]=4  # Noncoordinate winner is in the complement.
    loss=x.b.online.max_geometry_margin(z,c['coordinate_ids'][:999])
    assert float(loss.detach())==pytest.approx(torch.nn.functional.softplus(torch.tensor(4.)).item())
    loss.backward()
    assert z.grad[0]>0 and z.grad[c['coordinate_ids'][998]]<0
    context=c['contexts'][0]
    scores={i:-math.log(c['vocab_size']) for i in range(c['vocab_size'])}
    acquired=dict(token_ids=[c['coordinate_ids'][999]],full_scores=scores,
        processed_prompt_token_ids=context['processed_prompt_token_ids'],stop_reason='length',
        score_semantics='native_raw_logprobs_full_vocabulary',vocab_size=c['vocab_size'])
    measured=x.score_measure(c,context,acquired,'CPU')
    assert measured['margin']==0 and measured['rounded_zero_tie'] and not measured['literal_legal']
    assert measured['natural_recovery_credit'] is False
    del scores[0]
    with pytest.raises(AssertionError,match='full vocabulary'): x.score_measure(c,context,acquired,'CPU')


def test_contract_resigned_drift_and_source_boundary(prepared,tmp_path,monkeypatch):
    c=copy.deepcopy(prepared[0]); path=tmp_path/'manifest.json'
    old,bindings=x.historical()
    monkeypatch.setattr(x,'historical',lambda:(old,bindings))
    training=c['training']
    monkeypatch.setattr(x,'training_binding',lambda _:training)
    def check(value):
        write(path,value); return x.validate(path,current_source=False)
    check(c)
    for key in ['checkpoint','weight_identity','norm','input_path','label_path','generation']:
        bad=copy.deepcopy(c); bad[key]='changed'
        with pytest.raises(AssertionError): check(bad)
    for key in ['arms','phases','bounds','objective','gt_role','training']:
        bad=copy.deepcopy(c); bad[key]={}
        with pytest.raises(AssertionError): check(bad)
    with pytest.raises(AssertionError,match='release'):
        write(path,c); x.validate(path,require_release=True,current_source=False)
    # The unit entry calls the maintained clean-source guard before any model call.
    from src.artifacts.git_identity import SourceIdentityError
    c['source']={'schema':'obsolete'}; write(path,c)
    with pytest.raises(SourceIdentityError): x.validate(path)
    repo=tmp_path/'source'; repo.mkdir(); (repo/'runner.py').write_text('bound\n')
    for args in [['init','-q'],['add','runner.py'],['-c','user.name=CPU','-c','user.email=cpu@example.invalid','commit','-qm','CPU']]:
        subprocess.run(['git','-C',str(repo),*args],check=True,capture_output=True)
    identity=x.capture_source_identity(['runner.py'],root=repo)
    x.verify_source_identity(identity,required_paths=['runner.py'],root=repo)
    (repo/'runner.py').write_text('drift\n')
    with pytest.raises(SourceIdentityError): x.verify_source_identity(identity,required_paths=['runner.py'],root=repo)


def test_five_phase_callers_consumer_and_resigned_false_results(prepared,tmp_path,monkeypatch):
    from src.qwen import vllm_rollout
    c,q,inputs,requests,reports=prepared
    c=copy.deepcopy(c); c.update(source={'commit':'CPU-double'},native_released=True,
        execution_checkout=str(x.ROOT),output_root=str(tmp_path))
    output=tmp_path/'package'; output.mkdir(); path=tmp_path/'manifest.json'; write(path,c)
    expected_sha=x.b.sha(path)
    calls=[]; resets=[]; engines=[]; forwarded=[]; optimizer_records=[]
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
            forwarded.append(context['context_id'])
            # Actual last-logit differentiable replay with a history-sensitive CPU double.
            adjustment=(self.lora_weight+self.input_delta+self.output_delta)*(1+context['offset']*.1)
            return SimpleNamespace(logits=(self.mask*adjustment).view(1,1,-1))
    def compose(checkpoint,evaluation):
        assert str(checkpoint)==c['checkpoint'] and evaluation is False
        assert not engines or engines[-1].closed
        model=Model(); resets.append([float(t.detach()) for t in model.parameters()])
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
            assert kwargs['max_num_seqs']==1 and kwargs['max_model_len']==4456
            assert kwargs['kv_cache_memory_bytes']==2*1024**3 and kwargs['timeout']==1800
            assert not engines or engines[-1].closed
            self.identity=kwargs['identity']; self.receipts=[]; self.closed=False; self.score_index=0; self.natural_index=0
            self._process=SimpleNamespace(is_alive=lambda:not self.closed)
            self.arm=(['anchor']+x.ARMS)[len(engines)]
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
            legal=(self.score_index%2==1) if self.arm=='anchor' else (self.score_index!=4 if self.arm=='R-single' else True)
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
        x.native(path,output,expected_sha,'anchor')
        x.train(path,output,expected_sha,'R-single')
        x.native(path,output,expected_sha,'R-single')
        x.train(path,output,expected_sha,'R-multiple')
        x.native(path,output,expected_sha,'R-multiple')
        result=x.readback(path,output,expected_sha)
    finally:
        torch.set_num_threads(threads)
    assert resets[0]==resets[1] and len(optimizer_records)==2
    assert [g['lr'] for g in optimizer_records[0].param_groups]==[1e-5,5e-6]
    assert [int(v['step']) for v in optimizer_records[0].state.values()]==[16]*3
    assert len(forwarded)==222 and len(calls)==84 and all(e.closed for e in engines)
    assert result['counters']==dict(optimizer_steps=32,training_replays=192,HF_diagnostics=30,
        score_requests=30,continuation_requests=54,requests=84,generated_tokens=result['counters']['generated_tokens'])
    assert forwarded[:10]==[z['context_id'] for z in c['contexts']]
    assert forwarded[10:106]==[z['context_id'] for z in x.views(c,'R-single')]*16
    assert forwarded[116:212]==[z['context_id'] for z in x.views(c,'R-multiple')]*16
    assert result['conditional']['R-single']['7511-626/original']['baseline_illegal_denominator']==1
    assert result['conditional']['R-single']['7511-626/training_neighbor']['baseline_legal_denominator']==1
    assert result['conditional']['R-single']['7511-626/training_neighbor']['baseline_illegal_denominator']==1
    assert result['conditional_natural_recovery_credit'] is False
    original_complete=x.b.load(output/'complete.json')
    bad=copy.deepcopy(original_complete); bad['conditional']['R-single']['7511-626/original']['baseline_illegal_denominator']=99
    write(output/'complete.json',bad)
    with pytest.raises(AssertionError,match='re-signed false terminal'): x.readback(path,output,expected_sha)
    write(output/'complete.json',original_complete)

    def resign(directory,name,value):
        write(directory/name,value)
        receipt=x.b.load(directory/'complete.json'); receipt['artifacts'][name]=x.b.sha(directory/name)
        write(directory/'complete.json',receipt)
    directory=output/'native-R-single'; name='score-00.json'; original=x.b.load(directory/name)
    for change in ['literal_legal','natural_recovery_credit']:
        bad=copy.deepcopy(original); bad['measurement'][change]=not bad['measurement'][change]; resign(directory,name,bad)
        with pytest.raises(AssertionError,match='false conditional'): x.readback(path,output,expected_sha)
    resign(directory,name,original)
    name='natural-7511.json'; original=x.b.load(directory/name)
    bad=copy.deepcopy(original); bad['measurement']['ids']['category']['retained'].append(12345); resign(directory,name,bad)
    with pytest.raises(AssertionError,match='false natural credit'): x.readback(path,output,expected_sha)
    resign(directory,name,original)
    receipt=x.b.load(directory/'complete.json')
    for field,value in [('arm','R-multiple'),('checkpoint',str(output/'train-R-multiple/checkpoint-16')),('weight_identity','wrong'),('counters',{})]:
        bad=copy.deepcopy(receipt); bad[field]=value; write(directory/'complete.json',bad)
        with pytest.raises(AssertionError): x.readback(path,output,expected_sha)
    write(directory/'complete.json',receipt)
    directory=output/'train-R-multiple'; name='update-01.json'; original=x.b.load(directory/name)
    for change in ['context_id','weight','loss']:
        bad=copy.deepcopy(original); bad['terms'][0][change]='swapped' if change=='context_id' else 99; resign(directory,name,bad)
        with pytest.raises((AssertionError,TypeError)): x.readback(path,output,expected_sha)
    resign(directory,name,original)
    receipt=x.b.load(directory/'complete.json'); bad=copy.deepcopy(receipt); bad['fresh_optimizer']=False
    write(directory/'complete.json',bad)
    with pytest.raises(AssertionError): x.readback(path,output,expected_sha)
    write(directory/'complete.json',receipt)
    x.readback(path,output,expected_sha)


def test_phase_order_denominators_and_no_error_replacement(prepared):
    c=prepared[0]
    base=[dict(literal_legal=True,margin=0) for _ in c['contexts']]
    summary=x.contrast(c,dict(anchor=base,**{a:base for a in x.ARMS}))
    assert summary['original_error_contrast_present'] is False
    assert all(v['baseline_illegal_denominator']==0 and v['repaired']==[] for a in x.ARMS for v in summary[a].values())
    assert x.BOUNDS['generated_tokens']==30+54*3084 and x.BOUNDS['training_replays']==32*6
    commands=x.phase_commands('manifest','output')
    assert [c[3] for c in commands]==['native','train','native','train','native']
    assert [c[-1] for c in commands]==['anchor','R-single','R-single','R-multiple','R-multiple']
