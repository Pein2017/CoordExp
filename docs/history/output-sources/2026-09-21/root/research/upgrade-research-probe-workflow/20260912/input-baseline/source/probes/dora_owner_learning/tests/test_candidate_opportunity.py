"""Retained CPU entry fixture and decision-changing counterexamples."""
import copy
import json
from pathlib import Path

import pytest
from tokenizers import Tokenizer

from probes.dora_owner_learning.candidate_opportunity import (
    case_reduction, compare, digest, file_hash, run, score, validate_sample,
)


ROOT = Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration')
PLAN = ROOT/'2026-09-06-ce-controls-rloo-successor/ce-rloo-v1/rloo/round-1/plan.json'
GREEDY = ROOT/'2026-09-05-sft256-dev128-baseline/natural-eval-v1/qwen3-vl-2b-sft256-source-train256-natural-v1'
EID = 'coco2017_train_000000000368'


@pytest.fixture(scope='module')
def retained():
    plan=json.loads(PLAN.read_text())
    group=next(g for g in plan['population']['groups'] if g['example_id']==EID)
    shard=json.loads(Path(plan['sources']['rollout_artifacts'][0]['path']).read_text())
    row=next(r for r in shard['rollouts'] if r['example_id']==EID and r['seed']==group['actions'][0]['seed'])
    baseline=next(json.loads(l) for l in (GREEDY/'gt_vs_pred.jsonl').open() if json.loads(l)['row_id']==EID)
    tokenizer=Tokenizer.from_file(str(Path(plan['model']['base_model_path'])/'tokenizer.json'))
    return row,group['actions'][0],group,shard['prompt_metadata'][EID],baseline,tokenizer,3084


def test_retained_saved_entry_and_immutable_reload(tmp_path):
    result=run(PLAN,GREEDY,tmp_path/'result',[EID])
    saved=json.loads((tmp_path/'result/summary.json').read_text())
    cases=json.loads((tmp_path/'result/cases.json').read_text())
    assert result==saved
    assert saved['cases_sha256']==file_hash(tmp_path/'result/cases.json')
    assert saved['scope']=='declared_cpu_subset' and saved['samples']==4
    assert cases[0]['greedy']['50']['tp']==12
    assert [s['50']['tp'] for s in cases[0]['samples']]==[11,12,13,9]
    assert saved['witnesses']['strong_joint_witness']['images']==1
    assert saved['witnesses']['strong_joint_witness']['samples']==1
    before=file_hash(tmp_path/'result/summary.json')
    with pytest.raises(Exception,match='already exists'):
        run(PLAN,GREEDY,tmp_path/'result',[EID])
    assert file_hash(tmp_path/'result/summary.json')==before


@pytest.mark.parametrize('corruption', ['tokens','text','parser','reward','prompt','media','terminal','gt'])
def test_real_saved_sample_rejects_corruption(retained, tmp_path, corruption):
    row,action,group,meta,baseline,tokenizer,cap=retained
    # Actual retained bytes are saved/reloaded before exercising parser and matcher.
    fixture=tmp_path/'fixture.json'
    fixture.write_text(json.dumps([row,action,group,meta,baseline]))
    r,a,g,m,b=json.loads(fixture.read_text())
    assert validate_sample(r,a,g,m,b,tokenizer,cap)[0]['50']['tp']==11
    if corruption=='tokens': r['generated_token_ids'][0]=0
    elif corruption=='text': r['generated_text']+='x'
    elif corruption=='parser':
        r['predictions']['predictions'][0]['bbox'][0]+=1
        a['parser_evidence_sha256']=digest(r['predictions'])
    elif corruption=='reward': a['matching']['matched_owner_count']+=1
    elif corruption=='prompt': r['prompt_token_ids'][0]=0
    elif corruption=='media': r['executed_media_sha256']='0'*64
    elif corruption=='terminal': a['action_token_count']+=1
    else: b['gt'][0]['bbox']=[900,900,999,999]
    with pytest.raises(ValueError): validate_sample(r,a,g,m,b,tokenizer,cap)


def test_source_hash_corruption_rejected_before_reduction(tmp_path):
    plan=json.loads(PLAN.read_text())
    plan['sources']['rollout_artifacts'][0]['sha256']='0'*64
    plan['content_sha256']=digest({k:v for k,v in plan.items() if k!='content_sha256'})
    corrupted=tmp_path/'plan.json'; corrupted.write_text(json.dumps(plan))
    with pytest.raises(ValueError,match='source hash changed'):
        run(corrupted,GREEDY,tmp_path/'result',[EID])
    assert not (tmp_path/'result/summary.json').exists()


def card(owners, seed, *, repeated=False):
    boxes=[[0,0,100,100],[200,0,300,100],[400,0,500,100]]
    gt=[dict(object_id=str(i),description='person',bbox=b) for i,b in enumerate(boxes)]
    pred=[dict(description='person',bbox=boxes[i]) for i in owners]
    if repeated: pred.append(copy.deepcopy(pred[0]))
    return score(dict(row_id='synthetic',gt=gt,pred=pred,image_width=999,image_height=999,
                      dropped_prediction_count=0),seed=seed,length=40,stop='im_end')


def test_union_gain_is_not_complete_output_gain():
    result=case_reduction(card([0],-1),[card([1],1),card([0],2)])
    assert result['union_gain_without_single_tp_improvement']
    assert result['union_owners']==['0','1']
    assert not any(s['comparison']['net_owner_improvement'] for s in result['samples'])


def test_more_tp_with_old_owner_loss_is_not_preserving():
    change=compare(card([1,2],1),card([0],-1))
    assert change['net_owner_improvement'] and change['lost']==['0']
    assert not change['owner_preserving_improvement'] and not change['strong_joint_witness']


def test_repeat_and_fp_burden_prevent_strong_witness():
    baseline=card([0],-1)
    sample=card([0,1],1,repeated=True)
    assert sample['strict_repeats']==1 and sample['50']['fp']==1
    change=compare(sample,baseline)
    assert change['owner_preserving_improvement'] and not change['strong_joint_witness']
    assert compare(card([0,1],1),baseline)['strong_joint_witness']
