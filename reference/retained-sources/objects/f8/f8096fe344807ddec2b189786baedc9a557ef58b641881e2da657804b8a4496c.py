import copy
import json

import pytest

from probes.dora_owner_learning import selective_preservation_strong_eval as evaluator


def admission_fixture(tmp_path,monkeypatch):
    train=tmp_path/'training';train.mkdir();prior=tmp_path/'prior.json';prior.write_text('{}')
    monkeypatch.setattr(evaluator,'TRAINING',train);monkeypatch.setattr(evaluator,'PRIOR_RECEIPT',prior)
    monkeypatch.setattr(evaluator,'PRIOR_RECEIPT_SHA',evaluator.file_hash(prior));monkeypatch.setattr(evaluator,'validate_prior_path',lambda r:None)
    fixture=dict(status='passed',support_gradient_ratio=10,old_two_unchanged=True,compensation_and_clip_checks=True)
    smoke=dict(status='passed',updates=[1,2],initial_KL_zero=True,initial_reference_count=50,
        step2_max_KL_by_rank={str(i):1e-5 for i in range(8)},reduced_gradient_adapter_optimizer_rank_identity=True,frozen_bytes_unchanged=True,
        step1_full_vocab_max_abs_errors={'368':100.,'7116':100.},step1_top1_rank_equal=False)
    def seal():
        r=dict(numerical_admission=evaluator.ADMISSION,prior_path_proof={'path':str(prior),'sha256':evaluator.file_hash(prior)})
        for key,path,value in [('coefficient_fixture',train/'coefficient-fixture.json',fixture),('two_step',train/'two-step-smoke.json',smoke)]:
            path.write_text(json.dumps(value));r[key]={'path':str(path),'sha256':evaluator.file_hash(path)}
        return r
    return fixture,smoke,seal


def test_path_reuse_does_not_require_cross_coefficient_model_equality(tmp_path,monkeypatch):
    fixture,smoke,seal=admission_fixture(tmp_path,monkeypatch)
    # Deliberately huge cross-coefficient differences are not this admission gate.
    evaluator.validate_admission(seal())
    for obj,key,value in [(fixture,'support_gradient_ratio',1),(fixture,'old_two_unchanged',False),
        (fixture,'compensation_and_clip_checks',False),(smoke,'initial_KL_zero',False),(smoke,'initial_reference_count',49),
        (smoke,'step2_max_KL_by_rank',{str(i):0. for i in range(8)}),(smoke,'frozen_bytes_unchanged',False)]:
        old=obj[key];obj[key]=value
        with pytest.raises(ValueError):evaluator.validate_admission(seal())
        obj[key]=old
    r=seal();r['numerical_admission']='serial_dense48_global_gradient_v1'
    with pytest.raises(ValueError):evaluator.validate_admission(r)


def test_exact_support100_and_actual1104_uses(tmp_path,monkeypatch):
    train=tmp_path/'training';train.mkdir();inp=train/'inputs.json';inp.write_text('{}')
    monkeypatch.setattr(evaluator,'TRAINING',train);monkeypatch.setattr(evaluator,'validate_rank_completion',lambda r:None)
    monkeypatch.setattr(evaluator,'validate_admission',lambda r:None);monkeypatch.setattr(evaluator,'validate_source_receipt',lambda r,p:r['adapter'])
    ranks=[]
    for i in range(8):
        path=tmp_path/f'rank{i}.json';path.write_text(json.dumps({'counters':{'support_KL_trajectories':138}}));ranks.append({'path':str(path)})
    ids=list(map(str,range(48)));p=dict(support_image_ids=ids,support_action_states=5848)
    r=dict(schema_version='selective_preservation_strong.training.v1',status='completed',updates=23,stop_reason='fixed_steps',lambda_kl=10.,lambda_support_kl=100.,
        support_image_ids=ids,support_action_states=5848,adapter={'root':str(train/'adapter')},inputs_sha256=evaluator.file_hash(inp),rank_terminals=ranks,
        actual_support_uses=1104,final_scores={'case':{'A_vs_best_other_margin':-1.}})
    assert evaluator.validate_receipt(r,p)==r['adapter']
    for change in [dict(schema_version='selective_preservation_dense.training.v1'),dict(lambda_support_kl=10.),dict(lambda_kl=100.),
        dict(actual_support_uses=0),dict(status='provisional'),dict(updates=22),dict(support_image_ids=ids[:-1]),dict(support_action_states=1568),
        dict(adapter={'root':str(tmp_path/'old/adapter')})]:
        with pytest.raises(ValueError):evaluator.validate_receipt(dict(r,**change),p)
    (tmp_path/'rank7.json').write_text(json.dumps({'counters':{'support_KL_trajectories':0}}))
    with pytest.raises(ValueError,match='support-use'):evaluator.validate_receipt(r,p)


def test_exact_complete_four_shards_single_candidate():
    p={'packet':'frozen'};ts=[dict(shard=i,gpu=evaluator.GPUS[i],status='completed',manifest_sha256=evaluator.digest(p),continuations=evaluator.COUNTS[i],
        model_loads=1,score_forwards=2 if i==0 else 0,new_tokens=100,elapsed_seconds=1.,training_receipt_sha256='same') for i in range(4)]
    evaluator.require_complete_shards(ts,p)
    for change in [dict(status='failed'),dict(continuations=32),dict(score_forwards=0),dict(training_receipt_sha256='other')]:
        bad=copy.deepcopy(ts);bad[0].update(change)
        with pytest.raises(ValueError):evaluator.require_complete_shards(bad,p)
    with pytest.raises(ValueError):evaluator.require_complete_shards(ts[:3],p)
