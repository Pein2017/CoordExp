import copy
import json

import numpy as np
import pytest
from src.common.errors import ArtifactContractError

from probes.dora_owner_learning.branch_bridge import (
    conditional_accounting, entrance, publish, summarize_logits, validate_intervention,
)
from probes.source_rweak_row_cross.owner_row_robustness import branch, native_record, incidence


def case_fixture():
    a=dict(ids=[1,42,2,3,110,120,130,140,4],positions=[4,5,6,7])
    b=dict(ids=[1,42,2,3,115,125,135,145,4],positions=[4,5,6,7])
    return dict(case_id='coco2017_train_000000000368:2022537',A=a,B=b,prefix_ids=[8]*93,
                source_ids=[8]*93+b['ids']+[151645])


def test_entrance_is_before_x1_and_bound_to_source():
    c=case_fixture();e=entrance(c)
    assert e['action_index']==97 and len(e['state_ids'])==97
    assert e['A_id']==110 and e['B_id']==115
    assert e['state_ids']+[e['A_id']]==branch(c,'partial_A')['prefix']+branch(c,'partial_A')['forced']
    bad=copy.deepcopy(c);bad['prefix_ids'].append(8)
    with pytest.raises(ValueError,match='off-by-one'):entrance(bad)
    bad=copy.deepcopy(c);bad['source_ids'][0]=9
    with pytest.raises(ValueError,match='exact Source'):entrance(bad)
    bad=copy.deepcopy(c);bad['source_ids'][97]=114
    with pytest.raises(ValueError,match='A/B'):entrance(bad)


def test_full_vocabulary_rank_competitors_and_ties():
    v=np.array([3.,2.,2.,1.],dtype=np.float32);e=dict(A_id=2,B_id=3)
    s=summarize_logits(v,e)
    assert s['rank_min']==2 and s['rank_max']==3 and s['target_tie_count']==2
    assert s['top1_id']==s['best_other_id']==0
    assert s['A_vs_best_other_margin']==-1 and s['A_vs_B_margin']==1
    assert s['probability']==pytest.approx(np.exp(2)/np.exp(v.astype(float)).sum())
    v[:]=2;s=summarize_logits(v,e)
    assert s['top1_tie_count']==4 and s['rank_min']==1 and s['rank_max']==4
    with pytest.raises(ValueError,match='target/B'):summarize_logits(v,dict(A_id=2,B_id=2))


def test_intervention_rejects_budget_and_prefix_changes():
    c=case_fixture();j=branch(c,'partial_A')
    row=dict(arm='partial_A',prefix_ids=j['prefix'],forced_ids=j['forced'],remaining_budget=j['remaining'])
    validate_intervention(row,c)
    for field,value in [('prefix_ids',[]),('forced_ids',j['forced'][:-1]),('remaining_budget',j['remaining']+1)]:
        with pytest.raises(ValueError,match='altered intervention'):validate_intervention(dict(row,**{field:value}),c)


def test_occupied_publication_preserves_original(tmp_path):
    path=tmp_path/'receipt.json';publish(path,{'original':True})
    with pytest.raises(ArtifactContractError,match='path_already_exists'):publish(path,{'changed':True})
    assert json.loads(path.read_text())=={'original':True}


def test_current_row_not_conflated_with_later_owner_or_parser_drop():
    def text_row(box):
        return '<|object_ref_start|>person<|object_ref_end|><|box_start|>'+''.join(f'<|coord_{v}|>' for v in box)+'<|box_end|>'
    a=[10,20,30,40];b=[100,200,300,400]
    golden=dict(example_id='i',row_id='i',row_index=0,image_width=999,image_height=999,image_path='fixture',
        gt=[dict(object_id='a',description='person',bbox=a),dict(object_id='b',description='person',bbox=b)])
    case=dict(prefix_ids=[],boundary=0,owner='a',B_owner='b')
    for text,expected in [(text_row(a)+text_row(b),(True,False,True)),
                          (text_row(b)+text_row(a),(False,True,False)),
                          ('bad row<|box_end|>'+text_row(a),(False,True,False))]:
        parsed=native_record(text,{'row_id':'i'},golden,'im_end')
        row=dict(parsed=parsed,prefix_ids=[],prefix_text_length=0,text=text)
        got=conditional_accounting(row,case)['bridge_current_row']
        assert (got['A_direct'],got['A_later_suffix_direct'],got['B_free_direct'])==expected
