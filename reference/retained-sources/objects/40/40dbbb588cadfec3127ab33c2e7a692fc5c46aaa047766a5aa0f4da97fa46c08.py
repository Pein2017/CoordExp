import copy
import json

import pytest

from probes.dora_owner_learning.entrance_ce_eval import (
    guard_selection,owner_change,validate_natural,validate_receipt,native_record,file_hash,
    prepare,
)


def test_salted_numeric_guard_selection_is_frozen_and_order_independent():
    expected=[97,41,78,100,23,50,77,64,117,13,68,51,69,121,2,45]
    selected=guard_selection(list(range(1,129)),[1000,2000])
    assert [r['image_id'] for r in selected]==expected
    assert guard_selection([str(i) for i in reversed(range(1,129))],[1000,2000])==selected
    with pytest.raises(ValueError,match='collision'):guard_selection(list(range(1,129)),[97,1000])
    with pytest.raises(ValueError,match='dev128'):guard_selection(list(range(1,128))+[1],[1000,2000])


def test_owner_change_retains_identity_not_count_only():
    assert owner_change(['b','c'],['a','b'])==dict(gained=['c'],lost=['a'],retained=['b'])
    assert owner_change([],['a'])==dict(gained=[],lost=['a'],retained=[])


def test_occupied_evaluation_directory_is_preserved(tmp_path):
    sentinel=tmp_path/'existing';sentinel.write_text('keep')
    with pytest.raises(ValueError,match='occupied evaluation'):prepare(tmp_path)
    assert sentinel.read_text()=='keep'


class Tokenizer:
    def decode(self,ids,skip_special_tokens=False):
        return ''.join({1:'<|object_ref_start|>person<|object_ref_end|><|box_start|><|coord_10|><|coord_20|><|coord_30|><|coord_40|><|box_end|>',151645:'<|im_end|>'}[i] for i in ids)


def natural_fixture():
    tok=Tokenizer();golden=dict(example_id='i',row_id='i',row_index=0,image_width=999,image_height=999,image_path='fixture',
        gt=[dict(description='person',bbox=[10,20,30,40],object_id='a')])
    frozen=dict(example_id='i',case={'row_id':'i'},baseline=golden)
    text=tok.decode([1,151645]);row=dict(example_id='i',prefix_ids=[],forced_ids=[],action_ids=[1,151645],
        stop_reason='im_end',remaining_budget=3084,text=text,parsed=native_record(text,frozen['case'],golden,'im_end'))
    return tok,frozen,row


def test_natural_output_forcing_eos_cap_and_parser_are_fail_closed():
    tok,frozen,row=natural_fixture();assert validate_natural(row,frozen,tok)==row['parsed']
    for change in [dict(prefix_ids=[1]),dict(forced_ids=[1]),dict(remaining_budget=3085),dict(action_ids=[1]),
                   dict(action_ids=[1,151645,151645]),dict(action_ids=[1,151643,151645]),dict(stop_reason='length')]:
        with pytest.raises(ValueError):validate_natural(dict(row,**change),frozen,tok)
    bad=copy.deepcopy(row);bad['parsed']['gt'][0]['object_id']='wrong'
    with pytest.raises(ValueError,match='parser'):validate_natural(bad,frozen,tok)


def receipt_fixture(tmp_path):
    adapter=tmp_path/'adapter';adapter.mkdir();(adapter/'tensor').write_bytes(b'weights')
    embedding=tmp_path/'embedding';embedding.mkdir();(embedding/'tensor').write_bytes(b'embeddings')
    def identity(p):return dict(root=str(p),files=[dict(relative_path='tensor',sha256=file_hash(p/'tensor'))])
    ai,ei=identity(adapter),identity(embedding);source=dict(root='original-source',files=[])
    ent=dict(state_ids=[10,20],A_id=30,action_index=2)
    packet=dict(source_model=dict(source_embedding=ei,current_adapter=source),records=[dict(split='train',bridge_case={'case_id':'c'},
        entrance=ent,prompt_token_ids=[1,2])])
    receipt=dict(status='completed',adapter=ai,source_embedding=ei,source_adapter=source,updates=1,
        cases=[dict(case_id='c',state_ids=[10,20],prefix_token_ids=[10,20],target_token_id=30,action_index=2,prompt_token_ids=[1,2])],
        final_scores={'c':{'target_id':30}})
    return packet,receipt


def test_completed_checkpoint_barrier_and_exact_state_binding(tmp_path):
    packet,receipt=receipt_fixture(tmp_path);assert validate_receipt(receipt,packet)==receipt['adapter']
    with pytest.raises(ValueError,match='incomplete'):validate_receipt(dict(receipt,status='running'),packet)
    bad=copy.deepcopy(receipt);bad['cases'][0]['state_ids'].append(30)
    with pytest.raises(ValueError,match='entrance'):validate_receipt(bad,packet)
    bad=copy.deepcopy(receipt);bad['final_scores']={}
    with pytest.raises(ValueError,match='score coverage'):validate_receipt(bad,packet)
    (tmp_path/'adapter/tensor').write_bytes(b'changed')
    with pytest.raises(ValueError,match='payload bytes'):validate_receipt(receipt,packet)
