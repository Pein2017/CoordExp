import copy

import pytest

from probes.parallel_owner_research import transfer as t


def test_identity_projection_and_selection_exclude_all_exposure_kinds():
    evidence={'train_image_ids':[1,2], 'nested':{'image_id':3},
              'visual':'/x/val2017/000000000004.jpg','screen':'coco2017_train_000000000005',
              'object_id':6,'token_ids':[7,8]}
    assert t.image_ids(evidence)=={1,2,3,4,5}
    selected=t.select_ids(range(1,30),t.image_ids(evidence),10)
    assert not set(selected)&{1,2,3,4,5}
    assert selected==t.select_ids(reversed(range(1,30)),{5,4,3,2,1},10)
    with pytest.raises(ValueError,match='insufficient'):
        t.select_ids([1,2],{1},2)


def test_population_rejects_missing_duplicate_label_and_forced_credit():
    records=[{'example_id':'x'},{'example_id':'y'}]
    rows=[dict(example_id=x['example_id'],arm='C32',prefix_ids=[],forced_ids=[],remaining_budget=3084) for x in records]
    t.validate_population(rows,records,'C32')
    variants=[rows[:1],rows[::-1],rows[:1]*2]
    for key,value in [('arm','Stable50'),('forced_ids',[3]),('prefix_ids',[4]),('remaining_budget',3083)]:
        changed=copy.deepcopy(rows);changed[0][key]=value;variants.append(changed)
    for changed in variants:
        with pytest.raises(ValueError): t.validate_population(changed,records,'C32')


def test_first_divergent_row_boundary_and_absorbing_eos():
    assert t.first_row_prefix([10,11,99,10,12,99,151645],4,99)==([10,11,99,10,12,99],[151645])
    assert t.first_row_prefix([10,11,99,151645],3,99)==([10,11,99,151645],[])
    with pytest.raises(ValueError,match='neither complete row nor EOS'):
        t.first_row_prefix([10,11,99,10,12],4,99)


def test_old_exposed_input_rebasing_is_native_loader_compatible():
    from src.data.examples import raw_example_from_jsonl_row
    packet=t.read(t.PRIOR/'endpoint-preparation/packet.json')
    record=copy.deepcopy(packet['eval_records'][0])
    record['case']['input_record']['images']=[record['case']['image_path']]
    with pytest.raises(Exception,match='must be relative'):
        raw_example_from_jsonl_row(record['case']['input_record'],jsonl_path=t.NATIVE_SOURCE,row_number=1,raw_line='')
    t.rebase_case(record)
    raw=raw_example_from_jsonl_row(record['case']['input_record'],jsonl_path=t.NATIVE_SOURCE,row_number=1,raw_line='')
    assert str(raw.image.path)==record['case']['image_path']
    assert str(raw.example_id)==record['example_id']


def test_image_paired_uncertainty_preserves_gain_loss_concentration():
    def row(i,owners):
        n=len(owners)
        card={k:dict(tp=n,fp=0,fn=2-n,f1=2*n/(n+2),owners=owners) for k in ('50','60','80')}
        return dict(image_id=i,example_id=str(i),score=card)
    before=[row(1,['a']),row(2,['c'])]
    after=[row(1,['a','b']),row(2,[])]
    result=t.paired_summary(before,after)
    assert result['owner_counts']['50']==dict(gained=1,lost=1,retained=1)
    interval=result['paired_image_bootstrap']['intervals']['50']
    assert interval['delta_tp95']==[-2,2]
    assert interval['positive_images']==interval['negative_images']==1
    with pytest.raises(ValueError,match='unpaired'):
        t.paired_summary(before,after[::-1])
