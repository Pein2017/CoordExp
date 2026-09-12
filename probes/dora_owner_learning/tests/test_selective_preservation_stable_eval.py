import copy
import json
from pathlib import Path

import pytest

from probes.dora_owner_learning import selective_preservation_stable_eval as evaluator


def packet():return json.loads((evaluator.OUTPUT/'manifest.json').read_text())


def test_same384_new50_strata_and_old330_are_kept():
    p=packet();evaluator.validate_packet(p)
    assert len(p['old_outside330_ids'])==330 and len(p['targeted_stop_ids'])==3
    assert set(p['targeted_stop_ids'])<=set(p['old_outside330_ids'])
    assert {s:sum(r['split']==s for r in p['records']) for s in ['positive7','support50','remaining199','dev128']}==dict(positive7=7,support50=50,remaining199=199,dev128=128)
    for mutate in [lambda p:p['records'].pop(),lambda p:p['records'][0].update(example_id=p['records'][1]['example_id']),
        lambda p:p['old_outside330_ids'].pop(),lambda p:p['added_support_image_ids'].pop(),lambda p:p['support_image_ids'].append('529411')]:
        bad=copy.deepcopy(p);mutate(bad)
        with pytest.raises(ValueError):evaluator.validate_packet(bad)


def test_three_new_references_exact_source_actions_and_eos():
    p=packet();refs={r['image_id']:r for r in p['support_references']}
    assert [len(refs[str(i)]['action_ids']) for i in evaluator.TARGETED]==[10,95,81]
    for field in ('action_ids','prompt_token_ids'):
        bad=copy.deepcopy(p);ref=next(r for r in bad['support_references'] if r['image_id']=='73843');ref[field]=ref[field][:-1]
        with pytest.raises(ValueError):evaluator.validate_packet(bad)
    bad=copy.deepcopy(p);ref=next(r for r in bad['support_references'] if r['image_id']=='73843');ref['group']['image_path']='wrong'
    with pytest.raises(ValueError):evaluator.validate_packet(bad)


def test_wrong_old47_recipe_provisional_or_candidate_path_rejected(tmp_path,monkeypatch):
    p=packet();monkeypatch.setattr(evaluator,'TRAINING',tmp_path)
    receipt=dict(schema_version='selective_preservation_stable.training.v1',status='completed',numerical_admission=evaluator.ADMISSION,updates=81,stop_reason='fixed_steps',
        lambda_kl=10.,lambda_support_kl=100.,positive_image_ids=sorted(evaluator.TABLE),support_image_ids=p['support_image_ids'],local_KL_states=273,
        positive_trajectory_tokens=325,support_action_states=6024,total_cached_states=6297,adapter={'root':str(tmp_path/'adapter')},inputs_sha256='unread')
    for change in [dict(schema_version='selective_preservation_seven.training.v1'),dict(status='provisional'),dict(updates=23),dict(stop_reason='margin'),
        dict(support_action_states=5838),dict(total_cached_states=6111),dict(support_image_ids=p['support_image_ids'][:-1]),dict(adapter={'root':str(tmp_path/'old/adapter')})]:
        with pytest.raises(ValueError):evaluator.validate_receipt(dict(receipt,**change),p)


def test_complete_eight_shards_and_forced_context_fail_closed():
    p={'frozen':'p'};ts=[dict(shard=i,gpu=i,status='completed',manifest_sha256=evaluator.digest(p),continuations=48,model_loads=1,
        score_forwards=7 if i==0 else 0,new_tokens=100,elapsed_seconds=1.,training_receipt_sha256='same') for i in range(8)]
    evaluator.require_complete_shards(ts,p)
    for change in [dict(status='failed'),dict(continuations=47),dict(score_forwards=2),dict(training_receipt_sha256='other')]:
        bad=copy.deepcopy(ts);bad[0].update(change)
        with pytest.raises(ValueError):evaluator.require_complete_shards(bad,p)
    from probes.dora_owner_learning.tests.test_entrance_ce_eval import natural_fixture
    tok,frozen,row=natural_fixture()
    with pytest.raises(ValueError):evaluator.validate_natural(dict(row,prefix_ids=[1]),frozen,tok)
