import copy
import importlib.util
from pathlib import Path
import pytest
from tokenizers import Tokenizer

spec=importlib.util.spec_from_file_location('positive_release_runner',Path(__file__).with_name('runner.py'))
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


def test_native_first_fork_prefix_and_single_token_freeze():
    source=[151646,10,20,151645];sample=[151646,11,20,151645]
    v=dict(action_index=1,target_token_id=11,source_token_id=10,source_action_prefix_ids=[151646],source_action_prefix_sha256=runner.digest([151646]))
    b=runner.freeze_boundary(v,source,sample)
    assert b['extension_ids']==[151646,11] and b['forced_ids']==[11] and b['free_start']==2 and b['remaining_budget']==3082
    for changes in ({'action_index':2},{'target_token_id':12},{'source_action_prefix_ids':[151646,10]},{'source_token_id':99}):
        with pytest.raises(ValueError):runner.freeze_boundary({**v,**changes},source,sample)
    with pytest.raises(ValueError):runner.freeze_boundary(v,[151646,9,20,151645],[151646,9,11,151645])


@pytest.fixture(scope='module')
def tokenizer():
    return Tokenizer.from_file('/data/Qwen3-VL/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp-natural-adjacent/tokenizer.json')


def row(box):
    return '<|object_ref_start|>bottle<|object_ref_end|><|box_start|>'+''.join(f'<|coord_{n}|>' for n in box)+'<|box_end|>'


def geometry_fixture(tokenizer,order):
    boxes={'A':[100,100,200,200],'B':[500,500,600,600]}
    gold=dict(example_id='fixture',row_id='fixture',row_index=0,image_width=1000,image_height=1000,image_path='/fixture.jpg',
              gt=[dict(object_id=k,description='bottle',bbox=b) for k,b in boxes.items()])
    old_text=row(boxes['B'])+'<|im_end|>';old=runner.native_record(old_text,{'row_id':'fixture'},gold,'im_end')
    old_ids=tokenizer.encode(old_text,add_special_tokens=False).ids
    source=runner.score(old,seed=-1,length=len(old_ids),stop='im_end')
    text=''.join(row(boxes[k]) for k in order)+'<|im_end|>'
    ids=tokenizer.encode(text,add_special_tokens=False).ids
    parsed=runner.native_record(text,{'row_id':'fixture'},gold,'im_end');card=runner.score(parsed,seed=1,length=len(ids),stop='im_end')
    case=dict(example_id='fixture',owner='A',golden=gold)
    first=runner.first_row_evidence(ids,1,parsed,card,case,tokenizer)
    return source,card,first


def test_actual_first_row_global_assignment_not_later_target(tokenizer):
    source,card,first=geometry_fixture(tokenizer,['A','B'])
    assert first['complete'] and first['valid'] and first['full_global_target_assignment']['50']
    assert first['direct_target_iou']>.99
    assert runner.eligibility(source,card,first,'im_end')['candidate_eligible']
    source,card,first=geometry_fixture(tokenizer,['B','A'])
    assert 'A' in card['50']['owners'] and not first['full_global_target_assignment']['50']
    assert not runner.eligibility(source,card,first,'im_end')['candidate_eligible']


def test_lost_owner_is_rejected_despite_successful_entry_row(tokenizer):
    source,card,first=geometry_fixture(tokenizer,['A'])
    assert first['full_global_target_assignment']['50']
    assert 'lost_Source_IoU50_owner' in runner.eligibility(source,card,first,'im_end')['rejection_reasons']


@pytest.mark.parametrize('mutation,reason',[
    ('fp','annotation_relative_FP_increased'),('strict_repeats','strict_repeats_increased'),('parser_drops','parser_drops_increased')])
def test_burden_nonincrease_is_required(tokenizer,mutation,reason):
    source,card,first=geometry_fixture(tokenizer,['A','B']);changed=copy.deepcopy(card)
    if mutation=='fp':changed['50']['fp']+=1
    else:changed[mutation]+=1
    assert reason in runner.eligibility(source,changed,first,'im_end')['rejection_reasons']


def test_cap_and_lowest_seed_rule(tokenizer):
    source,card,first=geometry_fixture(tokenizer,['A','B'])
    assert 'not_natural_EOS' in runner.eligibility(source,card,first,'length')['rejection_reasons']
    rs=[dict(case_id='table',seed=s,variant_id=str(s),eligibility={'candidate_eligible':ok}) for s,ok in [(3,True),(2,True),(1,False),(4,True)]]
    assert runner.choose_lowest(rs)=={'table':'2'}


def test_occupied_packet_and_terminal_corruption(tmp_path):
    (tmp_path/'inputs.json').write_text('{}')
    with pytest.raises(ValueError):runner.prepare(tmp_path)
    with pytest.raises(ValueError):runner.checked_ids([151646,151645,2],'im_end')
