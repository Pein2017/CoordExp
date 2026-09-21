import copy
import json

import pytest

from probes.dora_owner_learning import selective_preservation_dense_eval as evaluator


def completion_fixture(tmp_path):
    ranks=[]
    for i,count in enumerate([216,168]+[144]*6):
        path=tmp_path/f'rank{i}.json';path.write_text(json.dumps(dict(rank=i,status='completed',updates=23,model_forwards=count)))
        ranks.append(dict(rank=i,path=str(path),sha256=evaluator.file_hash(path)))
    launch=tmp_path/'launcher.json';launch.write_text(json.dumps({'exit_code':0}))
    return dict(rank_terminals=ranks,launcher_exit={'path':str(launch),'sha256':evaluator.file_hash(launch)},global_model_forwards=1248)


def test_exact_eight_rank_completion_global_counts_and_launcher_exit(tmp_path):
    receipt=completion_fixture(tmp_path);evaluator.validate_global_completion(receipt)
    for change in [dict(rank_terminals=receipt['rank_terminals'][:7]),dict(rank_terminals=receipt['rank_terminals'][:7]+[receipt['rank_terminals'][0]]),dict(global_model_forwards=1247)]:
        with pytest.raises(ValueError):evaluator.validate_global_completion(dict(receipt,**change))
    path=tmp_path/'rank7.json';path.write_text(json.dumps(dict(rank=7,status='completed',updates=23,model_forwards=143)))
    receipt['rank_terminals'][7]['sha256']=evaluator.file_hash(path)
    with pytest.raises(ValueError,match='forward accounting'):evaluator.validate_global_completion(receipt)
    path.write_text(json.dumps(dict(rank=7,status='running',updates=23,model_forwards=144)));receipt['rank_terminals'][7]['sha256']=evaluator.file_hash(path)
    with pytest.raises(ValueError,match='incomplete'):evaluator.validate_global_completion(receipt)
    path.write_text(json.dumps(dict(rank=7,status='completed',updates=23,model_forwards=144)));receipt['rank_terminals'][7]['sha256']=evaluator.file_hash(path)
    launch=tmp_path/'launcher.json';launch.write_text(json.dumps({'exit_code':1}));receipt['launcher_exit']['sha256']=evaluator.file_hash(launch)
    with pytest.raises(ValueError,match='exit0'):evaluator.validate_global_completion(receipt)


def test_dense_recipe_not_old48_label_or_positive_margin_gate(tmp_path,monkeypatch):
    monkeypatch.setattr(evaluator,'ARM_ROOT',tmp_path);monkeypatch.setattr(evaluator,'validate_source_receipt',lambda r,p:r['adapter'])
    monkeypatch.setattr(evaluator,'validate_global_completion',lambda r:None)
    inputs=tmp_path/'training/inputs.json';inputs.parent.mkdir();inputs.write_text('{}');support=[str(i) for i in range(48)]
    packet=dict(support_image_ids=support,support_action_states=5848)
    receipt=dict(schema_version='selective_preservation_dense.training.v1',status='completed',updates=23,stop_reason='fixed_steps',
        lambda_kl=10.,lambda_support_kl=10.,support_image_ids=support,support_action_states=5848,
        inputs_sha256=evaluator.file_hash(inputs),adapter={'root':str(tmp_path/'training/adapter')},final_scores={'x':{'A_vs_best_other_margin':-5.}})
    assert evaluator.validate_receipt(receipt,packet)==receipt['adapter']
    for change in [dict(schema_version='selective_preservation_wide.training.v1'),dict(status='provisional'),dict(updates=22),dict(stop_reason='margin'),
                   dict(lambda_support_kl=1.),dict(lambda_kl=1.),dict(support_image_ids=support[:-1]),dict(support_action_states=1568),
                   dict(inputs_sha256='changed'),dict(adapter={'root':str(tmp_path/'other/adapter')})]:
        with pytest.raises(ValueError):evaluator.validate_receipt(dict(receipt,**change),packet)


def test_exact130_order_and_complete_single_candidate_shards():
    r=[dict(example_id=f'e{i}',split='train' if i<2 else 'guard' if i<18 else 'dev112') for i in range(130)]
    assert [len(s['example_ids']) for s in evaluator.expected_shards(r)]==[33,33,32,32]
    for mutate in [lambda x:x.pop(),lambda x:x[0].update(example_id='e1'),lambda x:x[1].update(split='guard')]:
        bad=copy.deepcopy(r);mutate(bad)
        with pytest.raises(ValueError):evaluator.expected_shards(bad)
    p={'frozen':'packet'};ts=[dict(shard=i,gpu=evaluator.GPUS[i],status='completed',manifest_sha256=evaluator.digest(p),continuations=evaluator.COUNTS[i],
        model_loads=1,score_forwards=2 if i==0 else 0,new_tokens=100,elapsed_seconds=1.,training_receipt_sha256='same') for i in range(4)]
    evaluator.require_complete_shards(ts,p)
    for change in [dict(status='failed'),dict(continuations=32),dict(score_forwards=0),dict(training_receipt_sha256='other')]:
        bad=copy.deepcopy(ts);bad[0].update(change)
        with pytest.raises(ValueError):evaluator.require_complete_shards(bad,p)
