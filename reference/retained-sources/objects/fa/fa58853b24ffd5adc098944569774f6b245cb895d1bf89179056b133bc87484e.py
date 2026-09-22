import copy
import json

import pytest

from probes.dora_owner_learning import selective_preservation_wide_eval as evaluator


def records():
    return [dict(example_id=f'e{i}',split='train' if i<2 else 'guard' if i<18 else 'dev112',baseline={'gt':[]}) for i in range(130)]


def test_exact130_round_robin_sizes_and_split_order():
    r=records();shards=evaluator.expected_shards(r)
    assert [len(s['example_ids']) for s in shards]==[33,33,32,32]
    assert shards[0]['example_ids']==[f'e{i}' for i in range(0,130,4)]
    for mutate in [lambda x:x.pop(),lambda x:x[0].update(example_id='e1'),lambda x:x[1].update(split='guard')]:
        bad=copy.deepcopy(r);mutate(bad)
        with pytest.raises(ValueError):evaluator.expected_shards(bad)


def test_packet_wrong_source_kl10_identity_and_missplit_fail_closed(tmp_path,monkeypatch):
    r=records();old18=dict(records=r[:18],source_model={'source':'original'},configs={'train':{},'guard':{}});old112=dict(records=r[18:],config={})
    monkeypatch.setattr(evaluator,'source_packets',lambda:(old18,old112));monkeypatch.setattr(evaluator,'KL10_ROOT',tmp_path)
    control=[dict(example_id=x['example_id'],split=x['split'],parsed={'gt':[]},score='frozen') for x in r]
    for path,data in [(tmp_path/'evaluation/execution/consumer.json',control[:18]),(tmp_path/'dev112/consumer.json',control[18:])]:
        path.parent.mkdir(parents=True);path.write_text(json.dumps(data))
    packet=dict(schema='selective_preservation_wide.eval.v1',arm='soft-preservation-wide31',records=r,shards=evaluator.expected_shards(r),
        source_model=old18['source_model'],configs=dict(old18['configs'],dev112={}),kl10_controls=control,source_files={})
    evaluator.validate_packet(packet)
    for mutate in [lambda p:p['source_model'].update(source='changed'),lambda p:p['kl10_controls'][0].update(score='changed'),
                   lambda p:p['kl10_controls'][0].update(split='guard'),lambda p:p['kl10_controls'][0].update(example_id='e1')]:
        bad=copy.deepcopy(packet);mutate(bad)
        with pytest.raises(ValueError):evaluator.validate_packet(bad)


def test_wide_recipe_exact31_fixed23_accepts_negative_margin(tmp_path,monkeypatch):
    monkeypatch.setattr(evaluator,'ARM_ROOT',tmp_path);monkeypatch.setattr(evaluator,'validate_source_receipt',lambda r,p:r['adapter'])
    inputs=tmp_path/'training/inputs.json';inputs.parent.mkdir();inputs.write_text('{}')
    support=[str(i) for i in range(31)];packet=dict(support_image_ids=support,support_preflight_sha256='frozen')
    receipt=dict(schema_version='selective_preservation_wide.training.v1',status='completed',updates=23,stop_reason='fixed_steps',
        lambda_kl=10.,lambda_support_kl=10.,support_image_ids=support,support_preflight_sha256='frozen',
        inputs_sha256=evaluator.file_hash(inputs),adapter={'root':str(tmp_path/'training/adapter')},final_scores={'x':{'A_vs_best_other_margin':-5.}})
    assert evaluator.validate_receipt(receipt,packet)==receipt['adapter']
    for change in [dict(schema_version='selective_preservation.training.v1'),dict(status='running'),dict(updates=22),dict(stop_reason='joint_margin'),
                   dict(lambda_support_kl=0),dict(lambda_kl=1),dict(support_image_ids=support[:-1]),dict(support_preflight_sha256='changed'),
                   dict(inputs_sha256='changed'),dict(adapter={'root':str(tmp_path/'old/adapter')})]:
        with pytest.raises(ValueError):evaluator.validate_receipt(dict(receipt,**change),packet)


def test_all_complete_shards_and_one_candidate_required():
    p={'packet':'identity'};ts=[dict(shard=i,gpu=evaluator.GPUS[i],status='completed',manifest_sha256=evaluator.digest(p),continuations=evaluator.COUNTS[i],
        model_loads=1,score_forwards=2 if i==0 else 0,new_tokens=100,elapsed_seconds=1.,training_receipt_sha256='same') for i in range(4)]
    evaluator.require_complete_shards(ts,p)
    for change in [dict(status='failed'),dict(continuations=32),dict(score_forwards=0),dict(new_tokens=101773),dict(training_receipt_sha256='other')]:
        bad=copy.deepcopy(ts);bad[0].update(change)
        with pytest.raises(ValueError):evaluator.require_complete_shards(bad,p)
    with pytest.raises(ValueError):evaluator.require_complete_shards(ts[:3],p)
