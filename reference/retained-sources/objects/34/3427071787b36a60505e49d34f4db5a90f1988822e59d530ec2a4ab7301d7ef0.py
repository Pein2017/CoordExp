"""Real retained round4 CPU fixture; deliberately not round1 scientific evidence."""
import copy
import json

import pytest

from probes.dora_owner_learning.round1_realization import (
    ROOT, SOURCE_ROOT, UPDATE_ROOT, owner_change, run, validate_case_identity,
    validate_identity, witness_join,
)
from probes.dora_owner_learning.candidate_opportunity import file_hash, rows

ROUND4=ROOT/'2026-09-06-ce-controls-rloo-successor/ce-rloo-v1/cold/source256-rloo-round4-train256-natural-v1'


def load(path):
    return json.loads(path.read_text())


def test_saved_historical_round4_fixture(tmp_path):
    result=run(ROUND4,tmp_path/'fixture',fixture_round4=True)
    saved=load(tmp_path/'fixture/summary.json')
    assert result==saved
    assert saved['scope']=='declared_historical_round4_cpu_fixture'
    assert saved['round']==4 and saved['images']==256
    assert [saved['source'][t]['tp'] for t in ('50','60','80')]==[1259,1190,908]
    assert [saved['post'][t]['tp'] for t in ('50','60','80')]==[1278,1209,923]
    assert saved['cases_sha256']==file_hash(tmp_path/'fixture/cases.json')
    change=saved['owner_changes']['50']
    assert change['gained']-change['lost']==19
    assert change['retained']+change['lost']==1259


def test_round4_cannot_be_called_round1(tmp_path):
    with pytest.raises(ValueError,match='wrong post checkpoint'):
        run(ROUND4,tmp_path/'not_round1')
    assert not (tmp_path/'not_round1/summary.json').exists()


@pytest.mark.parametrize('field',['checkpoint','embedding','policy'])
def test_saved_manifest_identity_corruption(field):
    source=load(SOURCE_ROOT/'run_manifest.json')
    post=load(ROUND4/'run_manifest.json')
    receipt=load(UPDATE_ROOT/'round-4/update/receipt.json')
    validate_identity(post,source,receipt)
    if field=='checkpoint': post['model_identity']['adapter']['adapter_path']+='/wrong'
    elif field=='embedding': post['model_identity']['embedding_delta']['identity']['delta_path']+='/wrong'
    else: post['generation_policy']['repetition_penalty']=1.1
    with pytest.raises(ValueError): validate_identity(post,source,receipt)


@pytest.mark.parametrize('field',['prompt','media','gt'])
def test_joined_real_case_corruption(field):
    source=rows(SOURCE_ROOT/'gt_vs_pred.jsonl')[0]
    eid=source['row_id']
    post=next(r for r in rows(ROUND4/'gt_vs_pred.jsonl') if r['row_id']==eid)
    oldimage=next(r for r in rows(SOURCE_ROOT/'image_plan.jsonl') if r['row_id']==eid)
    image=next(r for r in rows(ROUND4/'image_plan.jsonl') if r['row_id']==eid)
    oldprompt=next(r for r in load(SOURCE_ROOT/'run_manifest.json')['prompt_trace'] if r['row_id']==eid)
    prompt=next(r for r in load(ROUND4/'run_manifest.json')['prompt_trace'] if r['row_id']==eid)
    validate_case_identity(post,source,image,oldimage,prompt,oldprompt)
    if field=='prompt': prompt['backend_executed_prompt_token_ids_sha256']='0'*64
    elif field=='media': image['executed_media_sha256']='0'*64
    else: post['gt'][0]['bbox'][0]+=1
    with pytest.raises(ValueError): validate_case_identity(post,source,image,oldimage,prompt,oldprompt)


def test_owner_exchange_and_partial_witness_are_explicit():
    old={'50':{'owners':['a','b']}}
    post={'50':{'owners':['b','c','d']}}
    change=owner_change(post,old,'50')
    assert change=={'gained':['c','d'],'lost':['a'],'retained':['b'],'retention':.5}
    candidate={'samples':[{'seed':1,'50':{'owners':['a','b','c','e']},
        'comparison':{'gained':['c','e'],'net_owner_improvement':True,
                      'owner_preserving_improvement':True,'strong_joint_witness':True}}]}
    actions={1:{'advantage':-1.,'matching':{'matched_owner_refs':['a','b','c','e']}}}
    witness=witness_join(candidate,post,actions)[0]
    assert witness['appears_in_post']==['c'] and witness['absent_in_post']==['e']
    assert witness['advantage_sign']=='negative' and not witness['all_sample_owners_present']
    assert not witness['complete_sample_owner_set_reproduced']
