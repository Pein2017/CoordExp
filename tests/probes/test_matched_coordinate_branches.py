import copy
import json
from pathlib import Path
from uuid import UUID

import pytest

from probes import matched_coordinate_branches as m
from test_greedy_prefix_branching import frontend, saved, make_record, row


@pytest.fixture(scope='module')
def contract():
    return m.b.load(m.OUT/'cpu-02/contract.json')


def test_frozen_layout_and_historical_acceptance(contract, saved):
    requests = m.request_layout(contract)
    assert [r['site_id'] for r in requests] == [name for name,_ in m.FIXED for _ in range(6)]
    assert [r['condition'] for r in requests] == m.ORDER*4
    assert sum(r['budget'] for r in requests) == 59790
    for j,(name,bins) in enumerate(m.FIXED):
        group=requests[j*6:j*6+6];site=contract['sites'][j];raw=saved[site['image_id']]
        assert {r['prefix_tokens'] for r in group} == {site['position']+1}
        assert {r['budget'] for r in group} == {3084-site['position']-1}
        for r in group:
            assert r['extension'][:-1] == raw['token_ids'][:site['position']]
            assert r['processed_prompt_token_ids'] == raw['prompt_token_ids']+r['extension']
            assert r['supplied_token_id'] == contract['coordinate_ids'][bins[list(m.CONDITIONS).index(r['condition'])]]
        for a,z in [(0,5),(1,4),(2,3)]:
            assert group[a]['input_prefix_identity'] == group[z]['input_prefix_identity']
    receipt=contract['predecessor']['acceptance']
    assert receipt['sha256'] == 'cea0da67d3bf7f1469035af39d6522e734f7d4ac027c76dc2569311e0fa0d6ab'
    assert m.b.load(receipt['path'])['worker_candidate']['sha256'] == '63a9c1b6ae3bc3d0b48983f6a1cd7f06eee42977fbe54d9a8a3d23ab9b0abeec'
    assert m.b.load(m.PREDECESSOR/'native-terminal-candidate-01.json')['lead_native_accepted'] is False


def test_assisted_and_inherited_reemission_gets_no_new_unique_credit(frontend):
    boxes=[[10,10,100,100],[200,200,300,300],[400,400,500,500]]
    image=dict(image_id=1,cohort='human13',objects=[dict(coco_ann_id=j+1,desc='person',bbox_2d=box) for j,box in enumerate(boxes)])
    raw=make_record(frontend,''.join(row('person',box) for box in boxes[:2]+boxes))
    observations,_=m.b.online.observations(raw,frontend.tokenizer)
    site=dict(row=observations[1])
    result=m.measure(image,raw,site,frontend.tokenizer,raw['token_ids'][observations[1]['coordinate_positions'][0]])
    assert result['inherited']['category']==[1] and result['current_row']['category']==[2]
    assert result['later_free']['category']==[1,2,3]
    assert result['later_new_unique']['category']==[3]
    assert result['primary_semantic_vector']['later_free_category_owner_ids']==[1,2,3]
    # An assisted wrong-category geometric owner still cannot become a new physical owner later.
    raw=make_record(frontend,row('car',boxes[0])+row('person',boxes[0]))
    observations,_=m.b.online.observations(raw,frontend.tokenizer)
    result=m.measure(image,raw,dict(row=observations[0]),frontend.tokenizer,raw['token_ids'][observations[0]['coordinate_positions'][0]])
    assert result['current_row']['raw']==[1] and result['current_row']['category']==[]
    assert result['later_free']['category']==[1] and result['later_new_unique']['category']==[]


def test_stability_requires_both_compared_conditions_and_keeps_token_evidence_separate():
    artifacts=[]
    for condition in m.ORDER:
        artifacts.append(dict(request=dict(condition=condition,input_prefix_identity=condition),
            raw=dict(token_ids=[1],stop_reason='length'),measurements=dict(
                primary_semantic_vector=[1],whole=dict(raw=[1],category=[1]),
                current_row=dict(raw=[],category=[]),later_free=dict(raw=[1],category=[1]),
                later_new_unique=dict(raw=[1],category=[1]),burdens=dict(generated_tokens=1))))
    artifacts[-2]['raw']['token_ids']=[2]
    s=m.summarize(artifacts)
    assert s['stability']['A']['semantic'] and not s['stability']['A']['exact_tokens_and_stop']
    assert all(x['point'] is not None for x in s['comparisons'].values())
    artifacts[-2]['measurements']['whole']['raw']=[1,2]
    s=m.summarize(artifacts)
    assert s['comparisons']['C-A']['point']['whole']['raw'] is None
    assert s['comparisons']['C-A']['point']['whole']['category'] is not None
    artifacts[-2]['measurements']['primary_semantic_vector']=[2]
    s=m.summarize(artifacts)
    assert s['comparisons']['C-A']['point'] is None and s['comparisons']['C-B']['point'] is not None
    artifacts[-1]['measurements']['primary_semantic_vector']=[2]
    assert all(x['point'] is None for x in m.summarize(artifacts)['comparisons'].values())


def test_release_source_snapshot_and_payload_guards(tmp_path, contract, monkeypatch):
    c=copy.deepcopy(contract);path=tmp_path/'contract.json'
    def store():path.write_text(json.dumps(c))
    store()
    with pytest.raises(AssertionError,match='lead release'):m.validate_contract(path,require_release=True)
    with pytest.raises(ValueError,match='source identity'):m.validate_contract(path)
    c.update(native_released=True,execution_checkout=str(m.ROOT));store()
    with pytest.raises(ValueError,match='source identity'):m.validate_contract(path,require_release=True)
    # Historical source is retained, but cannot substitute for the new source gate.
    c['source']=m.b.load(m.PREDECESSOR/'released-contract-01.json')['source'];store()
    with pytest.raises(ValueError):m.validate_contract(path,require_release=True)
    monkeypatch.setattr(m,'verify_source_identity',lambda *a,**k:None)
    m.validate_contract(path,require_release=True)
    for key in ['weight_identity','input_path','checkpoint','norm']:
        before=c[key];c[key]='drift';store()
        with pytest.raises(AssertionError):m.validate_contract(path,require_release=True)
        c[key]=before
    c['requests'][0]['supplied_token_id']+=1;store()
    with pytest.raises(AssertionError,match='order/token/budget'):m.validate_contract(path,require_release=True)
    c['requests']=m.request_layout(c);c['requests'][0],c['requests'][1]=c['requests'][1],c['requests'][0];store()
    with pytest.raises(AssertionError,match='order/token/budget'):m.validate_contract(path,require_release=True)
    c['requests']=m.request_layout(c)
    # Mutate a small checkpoint binding to prove the real payload check has teeth.
    checkpoint=tmp_path/'checkpoint';checkpoint.mkdir();(checkpoint/'payload').write_text('bound')
    payload={'payload':m.b.sha(checkpoint/'payload')};(checkpoint/'identity.json').write_text(json.dumps(payload))
    c.update(checkpoint=str(checkpoint),checkpoint_files=payload);store()
    monkeypatch.setattr(m,'historical_bindings',lambda c:None)
    m.validate_contract(path,require_release=True)
    (checkpoint/'payload').write_text('changed')
    with pytest.raises(AssertionError,match='payload'):m.validate_contract(path,require_release=True)



def test_media_and_frontend_snapshot_fail_closed(contract, monkeypatch):
    original=m.b.online.native_batch
    def altered(q,item):
        from types import SimpleNamespace
        batch=original(q,item)
        return SimpleNamespace(prompt_token_ids=batch.prompt_token_ids,image_grids=batch.image_grids,media_sha256=['drift'])
    monkeypatch.setattr(m.b.online,'native_batch',altered)
    with pytest.raises(AssertionError):m.frontend(contract)
    changed=copy.deepcopy(contract);changed['tokenizer_sha256']='drift'
    with pytest.raises(AssertionError):m.frontend(changed)


def test_actual_24_request_producer_consumer_and_resigned_tampering(tmp_path,monkeypatch,contract,frontend,saved):
    from src.qwen import vllm_rollout
    c=copy.deepcopy(contract);c.update(source={'commit':'CPU-double'},native_released=True,
        output_root=str(tmp_path),execution_checkout=str(m.ROOT))
    path=tmp_path/'contract.json';path.write_text(json.dumps(c));calls=[];drift=[False]
    # Source/payload validation is tested above; the seam uses the real layout check.
    def validate(*args,**kwargs):
        assert c['requests']==m.request_layout(c)
        return c
    monkeypatch.setattr(m,'validate_contract',validate)
    class Engine:
        def __init__(self,**kwargs):
            assert kwargs['max_num_seqs']==1 and kwargs['max_logprobs']==-1 and kwargs['kv_cache_memory_bytes']==2*1024**3
            physical=dict(uuid='GPU-'+str(UUID(int=1)),pci_domain_id=0,pci_bus_id=1,pci_device_id=0)
            request=dict(schema='coordexp-vllm-device-1',rank=0,device=0,physical_token='0',
                parent=dict(pid=1,ppid=99,nspid='NSpid: 1',visibility=None,physical=physical))
            self.receipts=[];self.startup=dict(identity=c['weight_identity'],device=dict(requested=request,
                child=dict(pid=2,ppid=1,nspid='NSpid: 2'),inherited_visibility='0',effective_visibility='0',logical_device=0,physical=physical))
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def receipt(self,operation):
            return dict(operation=operation,identity=c['weight_identity'],coordinate_output_norm=dict(
                mode='off',identity=c['weight_identity'],calls=1,coordinate_tokens=1000,coordinate_ids=c['coordinate_ids'],
                first_call=dict(scaling_active=False,non_coordinate_unchanged=True)))
        def configure_coordinate_output_norm(self,mode,ids,**kwargs):
            assert mode=='off' and ids==c['coordinate_ids'] and kwargs['identity']==c['weight_identity']
            self.receipts.append(self.receipt('coordinate_output_norm'))
        def generate_exact(self,requests,**kwargs):
            assert len(requests)==len(kwargs['extensions'])==len(kwargs['budgets'])==1
            assert kwargs['full_scores'] is False and kwargs['identity']==c['weight_identity']
            index=len(calls);expected=c['requests'][index];site=c['sites'][index//6]
            assert kwargs['extensions'][0]==expected['extension'] and kwargs['budgets'][0]==expected['budget']
            raw=saved[site['image_id']];assert requests[0].request_id==raw['request_id']
            calls.append(kwargs);self.receipts.append(self.receipt('generate_exact'))
            suffix=raw['token_ids'][len(expected['extension']):3084];reason='length'
            if drift[0] and index==4:
                suffix=suffix[:5]+[frontend.tokenizer.convert_tokens_to_ids('<|im_end|>')];reason='im_end'
            return [dict(request_id=requests[0].request_id,token_ids=suffix,stop_reason=reason,
                full_scores=None,processed_prompt_token_ids=raw['prompt_token_ids']+kwargs['extensions'][0])]
    monkeypatch.setattr(vllm_rollout,'VllmDoraRollout',Engine)
    output=tmp_path/'stable';m.run(path,output,m.b.sha(path))
    result=m.readback(path,output,m.b.sha(path))
    assert len(calls)==24 and result['counters']['generated_tokens']==59790 and result['counters']['score_requests']==0
    assert all(x['semantic_stable'] for site in result['sites'] for x in site['comparisons'].values())
    complete_path=output/'complete.json';original=m.b.load(complete_path)
    directory=output/c['sites'][0]['site_id'];ledger_path=directory/'complete.json';ledger=m.b.load(ledger_path)
    artifact_path=directory/'A-1.json';value=m.b.load(artifact_path)
    def resign(changed_value, changed_ledger):
        artifact_path.write_text(json.dumps(changed_value))
        changed_ledger['artifacts']['A-1.json']=m.b.sha(artifact_path);ledger_path.write_text(json.dumps(changed_ledger))
        changed_complete=copy.deepcopy(original);changed_complete['sites'][0]=changed_ledger
        changed_complete['artifacts'][c['sites'][0]['site_id']+'/complete.json']=m.b.sha(ledger_path)
        complete_path.write_text(json.dumps(changed_complete))
    for field in ['supplied_token_id','condition','repeat','budget','request_index']:
        changed=copy.deepcopy(value);changed['request'][field]='tampered'
        resign(changed,copy.deepcopy(ledger))
        with pytest.raises(AssertionError,match='false request'):m.readback(path,output,m.b.sha(path))
    for part in ['later_free','later_new_unique']:
        changed=copy.deepcopy(value);changed['measurements'][part]['category']=[-1]
        resign(changed,copy.deepcopy(ledger))
        with pytest.raises(AssertionError,match='false request'):m.readback(path,output,m.b.sha(path))
    changed=copy.deepcopy(ledger);changed['stability']['A']['semantic']=False
    resign(copy.deepcopy(value),changed)
    with pytest.raises(AssertionError,match='false stability'):m.readback(path,output,m.b.sha(path))
    calls.clear();drift[0]=True;output=tmp_path/'unstable';m.run(path,output,m.b.sha(path))
    result=m.readback(path,output,m.b.sha(path));assert len(calls)==24
    assert result['sites'][0]['comparisons']['C-A']['point'] is None
    assert result['sites'][0]['comparisons']['C-B']['semantic_stable']
    assert all(x['semantic_stable'] for site in result['sites'][1:] for x in site['comparisons'].values())
    directory=output/c['sites'][0]['site_id'];lp=directory/'complete.json';changed=m.b.load(lp)
    changed['stability']['A']['semantic']=True;lp.write_text(json.dumps(changed))
    complete=m.b.load(output/'complete.json');complete['sites'][0]=changed
    complete['artifacts'][c['sites'][0]['site_id']+'/complete.json']=m.b.sha(lp)
    (output/'complete.json').write_text(json.dumps(complete))
    with pytest.raises(AssertionError,match='false stability'):m.readback(path,output,m.b.sha(path))
