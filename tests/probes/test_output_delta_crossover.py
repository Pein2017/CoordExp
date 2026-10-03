"""CPU serialization/readback and actual four native caller/final consumer doubles."""
import copy
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest
import torch
from safetensors.torch import load_file, save_file

from probes import output_delta_crossover as x


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


@pytest.fixture(scope='module')
def prepared():
    # Consume the already published CPU hybrids; never overwrite or prepare a second canonical candidate.
    c=x.validate(x.CANONICAL/'preparation-01/manifest.json',current_source=False)
    for item in c['weights'].values(): item['checkpoint']=str(Path(item['checkpoint']).resolve())
    q,inputs,requests,reports=x.frontend(c)
    assert q.model is None
    return c,q,inputs,requests,reports


def test_qualification_binds_absolute_canonical_inputs(prepared,tmp_path,monkeypatch):
    c,q,inputs,requests,reports=prepared
    monkeypatch.setattr(x,'capture_source_identity',lambda *a,**kw:dict(commit='CPU-double'))
    monkeypatch.setattr(x,'frontend',lambda _:(q,inputs,requests,reports))
    destination=tmp_path/'candidate.json'
    x.qualify(x.CANONICAL/'preparation-01/manifest.json',destination)
    candidate=x.b.load(destination)
    for arm,item in candidate['weights'].items():
        assert Path(item['checkpoint']).is_absolute()
        assert item==c['weights'][arm]
    assert candidate['native_released'] is False
    assert x.validate(destination,current_source=False)['weights']==c['weights']
    bad=copy.deepcopy(c); bad['weights']['R_M']['checkpoint']='outputs/wrong-execution-checkout-input'
    with pytest.raises(AssertionError,match='absolute canonical paths'):
        x.weight_binding(bad,tmp_path,'R_M')


def test_exact_assembly_readback_and_resigned_payload_sensitivity(prepared,tmp_path):
    c,*_=prepared
    destination=tmp_path/'R_M'
    item=x.assemble('R_M',c['parents'],c,destination)
    assert item['weight_identity']==c['weights']['R_M']['weight_identity']
    x.verify_arm(c,c['parents'],'R_M',item)
    for p in destination.rglob('*'):
        if p.is_dir(): p.chmod(0o755)
        else: p.chmod(0o644)
    destination.chmod(0o755)
    originals={name:(destination/name).read_bytes() for name in x.PAYLOADS+['identity.json','provenance.json']}
    rejected=[]

    def restore():
        for name,data in originals.items(): (destination/name).write_bytes(data)

    def resign():
        files={name:x.b.sha(destination/name) for name in x.PAYLOADS}
        write(destination/'identity.json',files)
        return dict(item,checkpoint_files=files,weight_identity=x.b.online.identity(files),
            provenance_sha256=x.b.sha(destination/'provenance.json'))

    cases=['input_instead_of_output','wrong_output_donor','mapping_order','body_payload','modified_delta',
        'copied_parent_identity','copied_parent_weight','wrong_arm','wrong_provenance_donor']
    for case in cases:
        restore(); bad=copy.deepcopy(item)
        if case in ['input_instead_of_output','wrong_output_donor','modified_delta']:
            body=load_file(str(Path(c['parents']['R']['checkpoint'])/x.DELTA))
            donor=load_file(str(Path(c['parents']['M']['checkpoint'])/x.DELTA))
            tensors=load_file(str(destination/x.DELTA))
            if case=='input_instead_of_output':
                tensors={'input_embed_delta':donor['input_embed_delta'],'output_embed_delta':body['output_embed_delta']}
            elif case=='wrong_output_donor': tensors['output_embed_delta']=body['output_embed_delta']
            else: tensors['output_embed_delta'][0,0]+=1
            save_file(tensors,str(destination/x.DELTA)); bad=resign()
        elif case=='mapping_order':
            data=x.b.load(destination/x.META)
            for key in ['token_strings','token_ids']: data[key][0],data[key][1]=data[key][1],data[key][0]
            write(destination/x.META,data); bad=resign()
        elif case=='body_payload':
            (destination/'adapter/model_card.json').write_bytes(originals['adapter/model_card.json']+b' ')
            bad=resign()
        elif case=='copied_parent_identity': write(destination/'identity.json',c['parents']['R']['checkpoint_files'])
        elif case=='copied_parent_weight': bad['weight_identity']=c['parents']['R']['weight_identity']
        elif case=='wrong_arm': bad['arm']='M_R'
        else:
            data=x.b.load(destination/'provenance.json'); data['output_donor']=c['parents']['R']
            write(destination/'provenance.json',data); bad=resign()
        with pytest.raises(AssertionError): x.verify_arm(c,c['parents'],'R_M',bad)
        rejected.append(case)
    restore(); x.verify_arm(c,c['parents'],'R_M',item)
    assert len(rejected)==9
    unreleased=tmp_path/'unreleased-absolute-inputs.json'; write(unreleased,c)
    with pytest.raises(AssertionError,match='exact lead release'):
        x.validate(unreleased,current_source=False,require_release=True)
    bad=copy.deepcopy(c); bad['edges'][0][1:]=list(reversed(bad['edges'][0][1:]))
    path=tmp_path/'wrong-direction.json'; write(path,bad)
    with pytest.raises(AssertionError,match='factorial direction'): x.validate(path,current_source=False)


def test_four_native_callers_final_consumer_and_resigned_false_evidence(prepared,tmp_path,monkeypatch):
    c,q,inputs,requests,reports=prepared
    c=copy.deepcopy(c); c.update(source={'commit':'CPU-double'},native_released=True,
        execution_checkout=str(x.ROOT),output_root=str(tmp_path))
    output=tmp_path/'package'; output.mkdir(); path=tmp_path/'manifest.json'; write(path,c)
    expected_sha=x.b.sha(path)
    # Only this probe's release/front-end/resource seams are doubled. Frozen predecessor globals stay intact.
    monkeypatch.setattr(x,'validate',lambda *a,**k:c)
    monkeypatch.setattr(x,'frontend',lambda _:(q,inputs,requests,reports))
    monkeypatch.setattr(x,'resources',lambda *a,**k:dict(seconds=.1,peak_rss_kib=1,
        reaped_children_peak_rss_kib=1,cuda_peak_allocated=None,cuda_peak_reserved=None))
    calls,engines=[],[]
    scores={i:-math.log(c['vocab_size']) for i in range(c['vocab_size'])}

    class Engine:
        def __init__(self,**kw):
            self.arm=x.ARMS[len(engines)]
            checkpoint,identity=x.weight_binding(c,output,self.arm)
            assert kw['checkpoint']==checkpoint and kw['identity']==identity
            assert kw['max_num_seqs']==1 and kw['max_model_len']==4456
            assert kw['kv_cache_memory_bytes']==2*1024**3 and kw['timeout']==1800
            assert kw['device']==kw['trainer_rank']==0 and kw['max_logprobs']==-1
            assert not engines or engines[-1].closed
            self.identity=identity; self.receipts=[]; self.closed=False
            self.score_index=0; self.natural_index=0
            self._process=SimpleNamespace(is_alive=lambda:not self.closed)
            physical=dict(uuid='GPU-'+str(UUID(int=1)),pci_domain_id=0,pci_bus_id=1,pci_device_id=0)
            request=dict(schema='coordexp-vllm-device-1',rank=0,device=0,physical_token='0',
                parent=dict(pid=1,ppid=99,nspid='NSpid: 1',visibility=None,physical=physical))
            self.startup=dict(identity=identity,device=dict(requested=request,
                child=dict(pid=2,ppid=1,nspid='NSpid: 2'),inherited_visibility='0',effective_visibility='0',
                logical_device=0,physical=physical))
            engines.append(self)
        def __enter__(self): return self
        def __exit__(self,*_): self.closed=True
        def receipt(self,operation):
            self.receipts.append(dict(operation=operation,identity=self.identity,coordinate_output_norm=dict(
                mode='off',identity=self.identity,coordinate_ids=c['coordinate_ids'],calls=1,
                first_call=dict(scaling_active=False,non_coordinate_unchanged=True))))
        def configure_coordinate_output_norm(self,mode,ids,**kw):
            assert mode=='off' and ids==c['coordinate_ids'] and kw['identity']==self.identity
            self.receipt('coordinate_output_norm')
        def generate_exact(self,request,**kw):
            context=c['contexts'][self.score_index]
            assert self.natural_index==0 and len(request)==1
            assert request[0].request_id==inputs[context['image_id']]['request_id']
            assert kw['budgets']==[1] and kw['full_scores'] is True and kw['identity']==self.identity
            assert kw['extensions']==[context['extension']]
            assert kw['chat_token_ids']==[next(z['unexpanded_chat_token_ids'] for z in reports if z['image_id']==context['image_id'])]
            legal=(self.score_index%2==0) if self.arm in ['R_R','M_R'] else True
            token=c['coordinate_ids'][0 if legal else 999]
            calls.append((self.arm,'score',context['context_id'])); self.score_index+=1
            self.receipt('generate_exact')
            return [dict(request_id=request[0].request_id,token_ids=[token],stop_reason='length',full_scores=scores,
                processed_prompt_token_ids=context['processed_prompt_token_ids'],vocab_size=c['vocab_size'],
                score_semantics='native_raw_logprobs_full_vocabulary')]
        def generate(self,request,**kw):
            assert len(request)==1 and self.score_index==10 and kw['budgets']==[3084]
            assert kw['identity']==self.identity and kw['eos_token_id']==c['eos_token_id']
            image=c['images'][self.natural_index]
            assert request[0].request_id==inputs[image]['request_id']
            tokens=[c['eos_token_id']]
            if image in [7511,351017] and self.arm!='R_R':
                raw=x.b.load(c['raw_paths'][str(image)])
                end=next(z['earlier_row']['positions'][-1]+1 for z in c['contexts'] if z['image_id']==image)
                tokens=raw['token_ids'][:end]+tokens
            calls.append((self.arm,'natural',image)); self.natural_index+=1; self.receipt('generate')
            return [SimpleNamespace(request_id=request[0].request_id,token_ids=tokens,stop_reason='im_end')]

    for arm in x.ARMS: x.native(path,output,expected_sha,arm,engine_factory=Engine)
    result=x.readback(path,output,expected_sha)
    assert x.readback(path,output,expected_sha)==result
    assert len(calls)==112 and len(engines)==4 and all(e.closed for e in engines)
    assert calls==[(arm,kind,item) for arm in x.ARMS for kind,items in
        [('score',[z['context_id'] for z in c['contexts']]),('natural',c['images'])] for item in items]
    assert result['counters']==dict(optimizer_steps=0,training_replays=0,HF_diagnostics=0,
        score_requests=40,continuation_requests=72,requests=112,generated_tokens=result['counters']['generated_tokens'])
    assert result['conditional_natural_recovery_credit'] is False and result['scientific_acceptance'] is False
    for arm in x.ARMS:
        summary=result['natural_summary'][arm]
        assert sum(summary['stop_counts'].values())==18
        assert summary['burdens']['generated_tokens']==sum(z['burdens']['generated_tokens'] for z in result['natural'][arm])
        for mode in ['raw','category']:
            assert summary['known_owner_ids'][mode]==[[z['image_id'],owner] for z in result['natural'][arm] for owner in sorted(set(z['ids'][mode]['retained']))]
    for name,before,after in x.EDGES:
        edge=result['natural_transitions'][name]
        assert (edge['before'],edge['after'])==(before,after)
        for key in result['natural'][before][0]['burdens']:
            expected=sum(a['burdens'][key]-z['burdens'][key] for z,a in zip(result['natural'][before],result['natural'][after]))
            assert edge['scalar_deltas'][key]==expected
    assert any(v!=0 for v in result['natural_transitions']['d_R']['scalar_deltas'].values())
    for key,value in result['interaction'].items():
        assert value==result['natural_transitions']['d_M']['scalar_deltas'][key]-result['natural_transitions']['d_R']['scalar_deltas'][key]

    original_terminal=x.b.load(output/'complete.json')
    terminal_bytes=(output/'complete.json').read_bytes()
    terminal_mutations=[lambda v:v['natural_transitions']['d_R'].update(before='R_M',after='R_R'),
        lambda v:v['natural_transitions']['d_R']['scalar_deltas'].update(generated_tokens=-v['natural_transitions']['d_R']['scalar_deltas']['generated_tokens']),
        lambda v:v['interaction'].update(generated_tokens=99),lambda v:v['counters'].update(requests=111)]
    for mutate in terminal_mutations:
        bad=copy.deepcopy(original_terminal); mutate(bad); write(output/'complete.json',bad)
        with pytest.raises(AssertionError,match='re-signed false terminal'): x.readback(path,output,expected_sha)
        (output/'complete.json').write_bytes(terminal_bytes)
    directory=output/'native-R_M'; receipt=x.b.load(directory/'complete.json')
    for field,value in [('arm','M_R'),('checkpoint',c['weights']['M_R']['checkpoint']),
        ('weight_identity',c['weights']['M_R']['weight_identity']),('source',{'commit':'false'}),
        ('counters',{}),('owned_child_absent',False)]:
        bad=copy.deepcopy(receipt); bad[field]=value; write(directory/'complete.json',bad)
        with pytest.raises(AssertionError): x.consume_native(c,output,'R_M',expected_sha,q,inputs)
        write(directory/'complete.json',receipt)
    def resign(name,value):
        write(directory/name,value)
        data=copy.deepcopy(receipt); data['artifacts'][name]=x.b.sha(directory/name)
        write(directory/'complete.json',data)
    for name,mutate in [('score-00.json',lambda v:v['measurement'].update(literal_legal=not v['measurement']['literal_legal'])),
        ('score-00.json',lambda v:v['measurement'].update(natural_recovery_credit=True)),
        ('natural-7511.json',lambda v:v['measurement']['ids']['category']['retained'].append(12345)),
        ('natural-7511.json',lambda v:v['measurement']['burdens'].update(generated_tokens=0))]:
        original=x.b.load(directory/name); original_bytes=(directory/name).read_bytes()
        bad=copy.deepcopy(original); mutate(bad); resign(name,bad)
        with pytest.raises(AssertionError,match='false'): x.consume_native(c,output,'R_M',expected_sha,q,inputs)
        (directory/name).write_bytes(original_bytes)
        write(directory/'complete.json',receipt)
    write(directory/'complete.json',receipt)
    assert x.readback(path,output,expected_sha)==result
    evidence=dict(status='CPU-candidate',real_model_loaded=False,real_native_launched=False,
        actual_caller_counters=result['counters'],four_fresh_serial_engine_doubles=True,
        exact_request_order=True,assembly_runtime_inspection=True,assembly_mutations_rejected=9,
        final_consumer_terminal_mutations_rejected=4,receipt_mutations_rejected=6,
        resigned_owner_score_length_mutations_rejected=4,reversed_factorial_direction_rejected=True,
        frozen_predecessor_globals_unchanged=True,native_startup_forwards=None,child_CUDA='unmeasured',
        first_hybrid_real_runtime='unresolved_until_release',scientific_acceptance=False)
    write(Path(os.environ.get('OUTPUT_DELTA_CPU_EVIDENCE',tmp_path/'cpu-evidence.json')),evidence)


def test_independent_factorial_sign_and_owner_sets(prepared):
    c,*_=prepared
    natural={arm:[dict(image_id=7511,burdens={'geometry_invalid':v,'generated_tokens':10+v},
        ids={'raw':{'retained':owners},'category':{'retained':owners}})] for arm,v,owners in
        [('R_R',2,[1,2]),('R_M',7,[2,3]),('M_R',11,[1,4]),('M_M',5,[2,4])]}
    scores={arm:[dict(literal_legal=True,margin=0.) for _ in c['contexts']] for arm in x.ARMS}
    result=x.factorial(c,scores,natural)
    assert [result['natural_transitions'][name]['scalar_deltas']['geometry_invalid'] for name,_,_ in x.EDGES]==[5,-6,9,-2,3]
    assert result['interaction']['geometry_invalid']==-11
    assert result['natural_transitions']['d_R']['images'][0]['owners']['category']==dict(gained=[3],lost=[1],retained=[2])
    assert result['natural_transitions']['d_M']['images'][0]['owners']['raw']==dict(gained=[2],lost=[1],retained=[4])
    assert all(group['baseline_illegal_denominator']==0 and group['repaired']==[]
        for edge in result['conditional'].values() for group in edge['groups'].values())
