import copy
import json
from uuid import UUID

import pytest

from probes import completed_row_crossover as x
from test_greedy_prefix_branching import frontend, make_record, row


@pytest.fixture(scope='module')
def contract():
    return x.b.load(x.OUT/'cpu-01/contract.json')


def test_actual_accepted_row_splice_and_source_separation(contract, frontend, tmp_path):
    c=contract; proof=x.row_boundary(c,frontend.tokenizer)
    assert c['row_boundary']==proof and proof['closing_token_id']==151649
    assert proof['start']==27 and proof['exclusive_end']==36
    requests=x.request_layout(c)
    assert requests==c['requests'] and [r['condition'] for r in requests]==x.ORDER
    assert {len(r['processed_prompt_token_ids']) for r in requests}=={1398}
    assert {r['prefix_tokens'] for r in requests}=={36} and {r['budget'] for r in requests}=={3048}
    assert sum(r['budget'] for r in requests)==18288
    raw=x.b.load(c['accepted_raw_paths']['C'][0])['raw']
    for r in requests:
        assert r['extension'][:31]==raw['token_ids'][:31] and r['extension'][35]==raw['token_ids'][35]
        assert r['extension'][31:35]==[c['coordinate_ids'][v] for v in x.CONDITIONS[r['condition']]]
        assert r['processed_prompt_token_ids']==raw['prompt_token_ids']+r['extension']
    for i,j in [(0,5),(1,4),(2,3)]:
        assert requests[i]['input_prefix_identity']==requests[j]['input_prefix_identity']
    old,paths,raws=x.accepted_history()
    assert paths==c['evidence_bindings'] and raws==c['accepted_raw_paths']
    assert c['source'] is None and c['predecessor']['historical_source_commit']==old['source']['commit']
    assert x.b.load(x.PREDECESSOR/'native-terminal-candidate-02.json')['lead_native_accepted'] is False
    changed=copy.deepcopy(c); path=tmp_path/'contract.json'
    path.write_text(json.dumps(changed))
    with pytest.raises(AssertionError,match='lead release'): x.validate_contract(path,require_release=True)
    with pytest.raises(ValueError,match='source identity'): x.validate_contract(path)
    changed['source']=old['source']; path.write_text(json.dumps(changed))
    with pytest.raises(ValueError): x.validate_contract(path)


def test_full_supplied_row_has_assisted_credit_only(frontend):
    boxes=[[10,10,100,100],[200,200,300,300],[400,400,500,500]]
    image=dict(image_id=1,cohort='human13',objects=[dict(coco_ann_id=i+1,desc='person',bbox_2d=box) for i,box in enumerate(boxes)])
    raw=make_record(frontend,''.join(row('person',box) for box in boxes[:2]+boxes))
    observations,_=x.b.online.observations(raw,frontend.tokenizer)
    result=x.m.measure(image,raw,dict(row=observations[1]),frontend.tokenizer,
                       raw['token_ids'][observations[1]['coordinate_positions'][0]])
    assert result['inherited']['category']==[1] and result['current_row']['category']==[2]
    assert result['first_row']['assisted_credit_only'] and result['first_row']['end']==18
    assert result['later_free']['category']==[1,2,3] and result['later_new_unique']['category']==[3]
    raw=make_record(frontend,row('car',boxes[0])+row('person',boxes[0]))
    observations,_=x.b.online.observations(raw,frontend.tokenizer)
    result=x.m.measure(image,raw,dict(row=observations[0]),frontend.tokenizer,raw['token_ids'][4])
    assert result['current_row']['raw']==[1] and result['current_row']['category']==[]
    assert result['later_free']['category']==[1] and result['later_new_unique']['category']==[]


def test_independent_pairwise_stability_and_supplementary_masking():
    artifacts=[dict(request=dict(condition=k,input_prefix_identity=k),raw=dict(token_ids=[1],stop_reason='length'),
        measurements=dict(primary_semantic_vector=[1],whole=dict(raw=[1],category=[1]),
            current_row=dict(raw=[],category=[]),later_free=dict(raw=[1],category=[1]),
            later_new_unique=dict(raw=[1],category=[1]),burdens=dict(generated_tokens=1))) for k in x.ORDER]
    artifacts[3]['raw']['token_ids']=[2]
    artifacts[3]['measurements']['whole']['raw']=[1,2]
    artifacts[3]['measurements']['burdens']['generated_tokens']=2
    summary=x.summarize(artifacts)
    assert summary['stability']['H']['semantic'] and not summary['stability']['H']['exact_tokens_and_stop']
    for name in ['C-H','B-H']:
        assert summary['comparisons'][name]['point']['whole']['raw'] is None
        assert summary['comparisons'][name]['point']['whole']['category'] is not None
        assert summary['comparisons'][name]['burden_delta_point']['generated_tokens'] is None
        assert summary['comparisons'][name]['burden_delta_ranges']['generated_tokens']==[0,1]
    assert summary['comparisons']['C-B']['point']['whole']['raw'] is not None
    artifacts[3]['measurements']['primary_semantic_vector']=[2]
    summary=x.summarize(artifacts)
    assert summary['comparisons']['C-H']['point'] is None and summary['comparisons']['B-H']['point'] is None
    assert summary['comparisons']['C-B']['point'] is not None
    artifacts[5]['measurements']['primary_semantic_vector']=[2]
    assert all(v['point'] is None for v in x.summarize(artifacts)['comparisons'].values())


def test_contract_rejects_resigned_rows_order_and_frozen_inputs(contract,tmp_path,monkeypatch):
    # Payload checks are unchanged; use a tiny checkpoint to exercise this caller's guard.
    old=x.b.load(x.PREDECESSOR/'released-contract-01.json'); expected=x.inherited_fields(old)
    checkpoint=tmp_path/'checkpoint'; checkpoint.mkdir(); (checkpoint/'payload').write_text('bound')
    payload={'payload':x.b.sha(checkpoint/'payload')}; (checkpoint/'identity.json').write_text(json.dumps(payload))
    expected.update(checkpoint=str(checkpoint),checkpoint_files=payload)
    c=copy.deepcopy(contract); c.update(expected)
    monkeypatch.setattr(x,'accepted_history',lambda:(old,contract['evidence_bindings'],contract['accepted_raw_paths']))
    monkeypatch.setattr(x,'inherited_fields',lambda old:expected)
    path=tmp_path/'contract.json'
    def check(value):
        path.write_text(json.dumps(value)); return x.validate_contract(path,current_source=False)
    check(c)
    for field in ['checkpoint','norm','input_path','weight_identity','tokenizer_sha256','generation']:
        changed=copy.deepcopy(c); changed[field]='drift'
        with pytest.raises(AssertionError,match=field): check(changed)
    changed=copy.deepcopy(c); changed['requests'][0]['extension'][35]+=1
    with pytest.raises(AssertionError,match='row/order/token/budget'): check(changed)
    changed=copy.deepcopy(c); changed['requests'][0],changed['requests'][1]=changed['requests'][1],changed['requests'][0]
    with pytest.raises(AssertionError,match='row/order/token/budget'): check(changed)
    changed=copy.deepcopy(c); changed['evidence_bindings'][str(x.PREDECESSOR/'lead-acceptance-01.json')]='drift'
    with pytest.raises(AssertionError): check(changed)
    (checkpoint/'payload').write_text('changed')
    with pytest.raises(AssertionError,match='payload'): check(c)


def test_actual_six_request_caller_consumer_and_resigned_tampering(contract,frontend,tmp_path,monkeypatch):
    from src.qwen import vllm_rollout
    c=copy.deepcopy(contract); c.update(source={'commit':'CPU-double'},native_released=True,
        output_root=str(tmp_path),execution_checkout=str(x.ROOT))
    path=tmp_path/'contract.json'; path.write_text(json.dumps(c)); calls=[]; drift=[False]
    def validate(*args,**kwargs):
        assert c['requests']==x.request_layout(c)
        return c
    monkeypatch.setattr(x,'validate_contract',validate)
    class Engine:
        def __init__(self,**kwargs):
            assert kwargs['max_num_seqs']==1 and kwargs['max_model_len']==4446
            assert kwargs['kv_cache_memory_bytes']==2*1024**3 and kwargs['timeout']==1200
            physical=dict(uuid='GPU-'+str(UUID(int=1)),pci_domain_id=0,pci_bus_id=1,pci_device_id=0)
            request=dict(schema='coordexp-vllm-device-1',rank=0,device=0,physical_token='0',
                parent=dict(pid=1,ppid=99,nspid='NSpid: 1',visibility=None,physical=physical))
            self.receipts=[]; self.startup=dict(identity=c['weight_identity'],device=dict(requested=request,
                child=dict(pid=2,ppid=1,nspid='NSpid: 2'),inherited_visibility='0',effective_visibility='0',logical_device=0,physical=physical))
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def receipt(self,operation):
            return dict(operation=operation,identity=c['weight_identity'],coordinate_output_norm=dict(
                mode='off',identity=c['weight_identity'],calls=1,coordinate_tokens=1000,coordinate_ids=c['coordinate_ids'],
                first_call=dict(scaling_active=False,non_coordinate_unchanged=True)))
        def configure_coordinate_output_norm(self,mode,ids,**kwargs):
            assert mode=='off' and ids==c['coordinate_ids']; self.receipts.append(self.receipt('coordinate_output_norm'))
        def generate_exact(self,requests,**kwargs):
            expected=c['requests'][len(calls)]
            assert len(requests)==len(kwargs['extensions'])==len(kwargs['budgets'])==1
            assert kwargs['extensions']==[expected['extension']] and kwargs['budgets']==[3048] and kwargs['full_scores'] is False
            assert kwargs['chat_token_ids']==[c['input_reports'][0]['unexpanded_chat_token_ids']]
            old=x.b.load(c['accepted_raw_paths']['B' if expected['condition']=='B' else 'C'][0])['raw']
            suffix=old['token_ids'][36:]; reason=old['stop_reason']
            if drift[0] and len(calls)==3:
                suffix=[frontend.tokenizer.convert_tokens_to_ids('<|im_end|>')]; reason='im_end'
            calls.append(expected); self.receipts.append(self.receipt('generate_exact'))
            return [dict(request_id=requests[0].request_id,token_ids=suffix,stop_reason=reason,
                full_scores=None,processed_prompt_token_ids=expected['processed_prompt_token_ids'])]
    monkeypatch.setattr(vllm_rollout,'VllmDoraRollout',Engine)
    output=tmp_path/'stable'; x.run(path,output,x.b.sha(path)); result=x.readback(path,output,x.b.sha(path))
    assert len(calls)==6 and [r['condition'] for r in calls]==x.ORDER
    assert result['counters']['score_requests']==0 and result['counters']['generated_tokens']<=18288
    directory=output/x.SITE; artifact_path=directory/'H-1.json'; ledger_path=directory/'complete.json'
    complete_path=output/'complete.json'; value=x.b.load(artifact_path); ledger=x.b.load(ledger_path); original=x.b.load(complete_path)
    assert value['credit_boundary']['supplied_row_natural_recovery_credit'] is False
    assert value['measurements']['first_row']['start']==27 and value['measurements']['first_row']['end']==36
    def resign(changed_value,changed_ledger):
        artifact_path.write_text(json.dumps(changed_value)); changed_ledger['artifacts']['H-1.json']=x.b.sha(artifact_path)
        ledger_path.write_text(json.dumps(changed_ledger)); complete=copy.deepcopy(original); complete['sites']=[changed_ledger]
        complete['artifacts'][x.SITE+'/complete.json']=x.b.sha(ledger_path); complete_path.write_text(json.dumps(complete))
    for field in ['condition','repeat','request_index','budget']:
        changed=copy.deepcopy(value); changed['request'][field]='tampered'; resign(changed,copy.deepcopy(ledger))
        with pytest.raises(AssertionError,match='false request'): x.readback(path,output,x.b.sha(path))
    changed=copy.deepcopy(value); changed['request']['extension'][35]+=1; resign(changed,copy.deepcopy(ledger))
    with pytest.raises(AssertionError,match='false request'): x.readback(path,output,x.b.sha(path))
    changed=copy.deepcopy(value); changed['raw']['token_ids'][31]+=1
    changed['raw']=x.b.online.seal(changed['raw'],changed['raw']['producer']); resign(changed,copy.deepcopy(ledger))
    with pytest.raises(AssertionError,match='false request'): x.readback(path,output,x.b.sha(path))
    changed=copy.deepcopy(value); changed['credit_boundary']['supplied_row_natural_recovery_credit']=True
    resign(changed,copy.deepcopy(ledger))
    with pytest.raises(AssertionError,match='false request'): x.readback(path,output,x.b.sha(path))
    changed=copy.deepcopy(value); changed['measurements']['later_new_unique']['category']=[-1]
    resign(changed,copy.deepcopy(ledger))
    with pytest.raises(AssertionError,match='false request'): x.readback(path,output,x.b.sha(path))
    changed=copy.deepcopy(ledger); changed['stability']['H']['semantic']=False; resign(copy.deepcopy(value),changed)
    with pytest.raises(AssertionError,match='false stability'): x.readback(path,output,x.b.sha(path))
    calls.clear(); drift[0]=True; output=tmp_path/'unstable'; x.run(path,output,x.b.sha(path)); result=x.readback(path,output,x.b.sha(path))
    assert len(calls)==6
    assert result['sites'][0]['comparisons']['C-B']['point'] is not None
    assert result['sites'][0]['comparisons']['C-H']['point'] is None and result['sites'][0]['comparisons']['B-H']['point'] is None
    ledger_path=output/x.SITE/'complete.json'; changed=x.b.load(ledger_path)
    changed['stability']['H']['semantic']=True; ledger_path.write_text(json.dumps(changed))
    complete=x.b.load(output/'complete.json'); complete['sites']=[changed]
    complete['artifacts'][x.SITE+'/complete.json']=x.b.sha(ledger_path)
    (output/'complete.json').write_text(json.dumps(complete))
    with pytest.raises(AssertionError,match='false stability'): x.readback(path,output,x.b.sha(path))
