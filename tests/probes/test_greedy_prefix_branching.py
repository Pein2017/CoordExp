import copy
import json
import math
from pathlib import Path

import pytest

from probes import greedy_prefix_branching as b


@pytest.fixture(scope='module')
def frontend():
    return b.rows.frontend()


@pytest.fixture(scope='module')
def saved(frontend):
    return {i:b.load(next((b.OLD/'native-01/off').glob(f'rank-*/{i}.json'))) for i in b.IMAGES}


def test_declared_saved_cases_and_first_impossible_coordinate(frontend,saved):
    for i,positions in [(7511,[626,203]),(351017,[1507,31])]:
        sites,dispositions=b.select_sites(saved[i],frontend.tokenizer)
        assert [s['position'] for s in sites] == positions
        assert [s['slot'] for s in sites] == [0,0]
        assert sites[0]['emitted_bin'] == 999 and sites[0]['legal_range'] == [0,999]
        assert all(s['eligible'] for s in sites) and dispositions['malformed_or_censored'] == 1
        for site in sites:
            prefix,budget=b.branch_prefix(saved[i],site)
            sham,sham_budget=b.branch_prefix(saved[i],site,site['emitted_token_id'])
            assert sham == prefix+[saved[i]['token_ids'][site['position']]]
            assert len(prefix)+budget == len(sham)+sham_budget == 3084
            assert budget-sham_budget == 1
            assert site['causal_logits_position'] == len(saved[i]['prompt_token_ids'])+len(prefix)-1
        altered=copy.deepcopy(sites[0]);altered['prefix_identity']='wrong'
        with pytest.raises(AssertionError):b.branch_prefix(saved[i],altered)


def make_record(frontend,text):
    ids=frontend.tokenizer.encode(text,add_special_tokens=False)
    return dict(image_id=1,request_id='cpu',crop=[0,0,1000,1000],width=1000,height=1000,
        arm='greedy',prompt_token_ids=[42,43],token_ids=ids,text=frontend.tokenizer.decode(ids,skip_special_tokens=False),
        generated_tokens=len(ids),stop_reason='im_end',raw_identity='cpu')


def row(name,box):
    return '<|object_ref_start|>'+name+'<|object_ref_end|><|box_start|>'+''.join(f'<|coord_{x}|>' for x in box)+'<|box_end|>'


def test_first_impossible_and_dedup_are_prediction_only(frontend):
    record=make_record(frontend,row('person',[10,20,10,40])*2)
    sites,_=b.select_sites(record,frontend.tokenizer)
    assert sites[0]['slot'] == 2 and sites[0]['legal_range'] == [11,1000]
    assert sites[1]['slot'] == 0 and sites[1]['row']['order'] == 1
    record=make_record(frontend,row('person',[999,20,999,40]))
    sites,_=b.select_sites(record,frontend.tokenizer)
    assert sites[0]['slot'] == 0 and not sites[1]['eligible']
    assert b.legal_range([5,999]) == (0,0)
    # A first complete invalid row repeated later is still a literal repeat.
    assert sites[0]['position'] < sites[0]['row']['coordinate_positions'][2]


def native_distribution():
    scores={t:-math.inf for t in range(1005)}
    for t,p in [(10,.5),(11,.2),(12,.2),(1004,.1)]:scores[t]=math.log(p)
    return scores


def test_full_native_ranking_denominator_ties_and_zero_probability():
    scores=native_distribution();site=dict(legal_range=[0,999],emitted_token_id=10)
    result=b.native_candidates(scores,list(range(1000)),site,1005)
    assert [x['token_id'] for x in result] == [11,12]
    region=b.score_region(scores,[10,11,12])
    assert region['mass'] == pytest.approx(.9)
    assert region['best_acceptable_minus_best_unacceptable'] == pytest.approx(math.log(5))
    assert b.score_region(scores,[])['mass'] == 0 and b.score_region(scores,[])['best_acceptable_minus_best_unacceptable'] is None
    assert b.score_region(scores,[0])['best_acceptable_minus_best_unacceptable'] == '-inf'
    for corrupt in [dict(scores,missing=0), {k:v for k,v in scores.items() if k!=0},
                    dict(scores,**{'0':float('nan')})]:
        with pytest.raises(AssertionError):b.native_candidates(corrupt,list(range(1000)),site,1005)
    for value in [float('nan'),float('inf')]:
        changed=dict(scores);changed[0]=value
        with pytest.raises(AssertionError):b.native_candidates(changed,list(range(1000)),site,1005)


def test_assisted_owner_is_separate_from_later_free_owner(frontend):
    boxes=[[10,10,100,100],[200,200,300,300],[400,400,500,500]]
    image=dict(image_id=1,cohort='human13',objects=[dict(coco_ann_id=j+1,desc='person',bbox_2d=box) for j,box in enumerate(boxes)])
    record=make_record(frontend,''.join(row('person',box) for box in boxes))
    observations,_=b.online.observations(record,frontend.tokenizer)
    site=dict(row=observations[1])
    out=b.evaluate_branch(image,record,site,frontend.tokenizer,record['token_ids'][observations[1]['coordinate_positions'][0]])
    assert out['whole']['category'] == [1,2,3]
    assert out['inherited']['category'] == [1]
    assert out['current_row']['category'] == [2]
    assert out['later_free']['category'] == [3]
    assert out['first_row']['assisted_credit_only'] and out['denominator_ids'] == [1,2,3]
    assert not out['annotation_unmatched_is_physical_negative']
    regions=b.owner_regions(image,'person',[])
    assert set(regions['per_owner']) == {'1','2','3'} and regions['union'] == sorted(set().union(*map(set,regions['per_owner'].values())))
    assert b.owner_regions(image,'unknown',[])['union'] == []


def synthetic_contract(tmp_path):
    checkpoint=tmp_path/'checkpoint';checkpoint.mkdir()
    (checkpoint/'weight').write_text('bound')
    payload={'weight':b.sha(checkpoint/'weight')}
    (checkpoint/'identity.json').write_text(json.dumps(payload))
    labels=tmp_path/'labels.json';labels.write_text(json.dumps([{'objects':[{}]*570}]+[{'objects':[]}]*17))
    return dict(schema='greedy-prefix-branching-v1',images=b.IMAGES,norm='off',branches=b.BRANCHES,
        generation=dict(temperature=0,top_p=1,top_k=-1,repetition_penalty=1,min_tokens=0,cap=3084,seed=92711),
        annotation_denominator=570,coordinate_ids=list(range(1000)),vocab_size=1005,source=None,native_released=False,bindings={str(labels):b.sha(labels)},
        checkpoint=str(checkpoint),checkpoint_files=payload,runtime={},label_path=str(labels),sites=[])


def test_real_contract_consumer_rejects_unreleased_or_unbound_source(tmp_path,monkeypatch):
    contract=synthetic_contract(tmp_path);path=tmp_path/'contract.json';path.write_text(json.dumps(contract))
    with pytest.raises(AssertionError,match='release'):b.validate_contract(path,require_release=True)
    with pytest.raises(ValueError,match='source identity'):b.validate_contract(path)
    monkeypatch.setattr(b,'verify_source_identity',lambda *a,**k:None)
    b.validate_contract(path)
    (Path(contract['checkpoint'])/'weight').write_text('drift')
    with pytest.raises(AssertionError):b.validate_contract(path)
    (Path(contract['checkpoint'])/'weight').write_text('bound')
    Path(contract['label_path']).write_text('[]')
    with pytest.raises(AssertionError):b.validate_contract(path)


def test_actual_run_entry_seals_branches_and_records_fidelity_hold(tmp_path,monkeypatch,frontend,saved):
    """Exercise the producer through final JSONs; runtime is CPU test double only."""
    from src.qwen import vllm_rollout
    c=b.load(b.OUT/'cpu-01/contract.json')
    # One finite site exercises all branches; a replay drift exercises HOLD without retries.
    c['sites']=[c['sites'][1]]
    c['source']={'commit':'cpu-contract-test'}
    monkeypatch.setattr(b,'validate_contract',lambda *a,**k:c)
    contract_path=tmp_path/'contract.json';contract_path.write_text(json.dumps(c))
    calls=[]
    class Engine:
        def __init__(self,**kwargs):
            assert kwargs['max_logprobs']==-1 and kwargs['max_num_seqs']==1
            from uuid import UUID
            physical=dict(uuid='GPU-'+str(UUID(int=1)),pci_domain_id=0,pci_bus_id=1,pci_device_id=0)
            request=dict(schema='coordexp-vllm-device-1',rank=0,device=0,physical_token='0',
                parent=dict(pid=1,ppid=99,nspid='NSpid: 1',visibility=None,physical=physical))
            self.receipts=[];self.startup=dict(identity=c['weight_identity'],device=dict(requested=request,
                child=dict(pid=2,ppid=1,nspid='NSpid: 2'),inherited_visibility='0',effective_visibility='0',logical_device=0,physical=physical))
        def receipt(self,operation):
            return dict(operation=operation,identity=c['weight_identity'],coordinate_output_norm=dict(
                mode='off',identity=c['weight_identity'],calls=1,coordinate_tokens=1000,coordinate_ids=c['coordinate_ids'],
                first_call=dict(scaling_active=False,non_coordinate_unchanged=True)))
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def configure_coordinate_output_norm(self,mode,ids,**kwargs):
            assert mode=='off' and ids==c['coordinate_ids']
            self.receipts.append(self.receipt('coordinate_output_norm'))
        def generate_exact(self,requests,**kwargs):
            calls.append(kwargs);self.receipts.append(self.receipt('generate_exact'))
            site=c['sites'][0];record=saved[site['image_id']];prefix=kwargs['extensions'][0]
            length=kwargs['budgets'][0]
            tokens=record['token_ids'][len(prefix):len(prefix)+length]
            if len(calls)==2 and drift[0]:tokens[0]=c['coordinate_ids'][0]
            scores=None
            if kwargs['full_scores']:
                emitted=site['emitted_token_id']
                scores={t:('-inf' if t!=emitted else 0.) for t in range(c['vocab_size'])}
            return [dict(request_id=requests[0].request_id,token_ids=tokens,stop_reason='length',
                full_scores=scores,score_semantics='native_raw_logprobs_full_vocabulary' if scores else None,processed_prompt_token_ids=record['prompt_token_ids']+prefix)]
    monkeypatch.setattr(vllm_rollout,'VllmDoraRollout',Engine)
    drift=[False]
    output=tmp_path/'normal';b.run(contract_path,output,b.sha(contract_path))
    complete=b.load(output/'complete.json')
    assert complete['sites'][0]['completed_branches']==b.BRANCHES
    assert complete['counters']['score_requests']==1 and complete['counters']['continuation_requests']==4
    assert [x['full_scores'] for x in calls]==[True,False,False,False,False]
    directory=output/c['sites'][0]['site_id']
    for name in b.BRANCHES:
        artifact=b.load(directory/(name+'.json'))
        assert artifact['raw']['producer']['source']=='cpu-contract-test'
        assert artifact['raw']['raw_identity'] != saved[7511]['raw_identity']
        assert artifact['budget']+artifact['prefix_tokens']==3084
        assert artifact['free_tokens'] <= artifact['budget'] and artifact['acquisition']['full_scores'] is None
    result=b.readback(contract_path,output,b.sha(contract_path))
    assert result['counters']['requests']==5 and result['annotation_denominator']==570
    # Re-sign the file transport hashes: the consumer must still reject false credit.
    artifact_path=directory/'alternative1.json';artifact=b.load(artifact_path)
    artifact['measurements']['later_free']['category']=[-1]
    artifact_path.write_text(json.dumps(artifact))
    ledger_path=directory/'complete.json';ledger=b.load(ledger_path)
    ledger['artifacts']['alternative1.json']=b.sha(artifact_path);ledger_path.write_text(json.dumps(ledger))
    complete['sites'][0]=ledger;complete['artifacts'][c['sites'][0]['site_id']+'/complete.json']=b.sha(ledger_path)
    (output/'complete.json').write_text(json.dumps(complete))
    with pytest.raises(AssertionError):b.readback(contract_path,output,b.sha(contract_path))
    calls.clear();drift[0]=True
    output=tmp_path/'drift';b.run(contract_path,output,b.sha(contract_path))
    complete=b.load(output/'complete.json')
    assert complete['status']=='HOLD_native_fidelity'
    assert len(calls)==3 and complete['sites'][0]['completed_branches']==b.BRANCHES[:2]
    assert not complete['sites'][0]['fidelity']['greedy_historical']
    assert not complete['sites'][0]['fidelity']['greedy_sham']
    result=b.readback(contract_path,output,b.sha(contract_path))
    assert result['status']=='HOLD_native_fidelity' and result['counters']['requests']==3
