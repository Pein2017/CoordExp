import copy
import json
from pathlib import Path

import pytest

from probes.dora_owner_learning import selective_preservation_seven_eval as evaluator


def frozen_packet():
    return json.loads((evaluator.OUTPUT/'manifest.json').read_text())


def test_all_seven_real_token_masks_entry_eos_and_conflict_removal():
    p=frozen_packet();assert len(p['positive_cases'])==7
    assert '529411' not in p['support_image_ids'] and len(p['support_image_ids'])==47
    assert sum(len(t['preservation_positions']) for t in p['trajectories'].values())==273
    for c in p['positive_cases']:
        t=p['trajectories'][c['case_id']];evaluator.validate_mask(c,t)
        wrong=copy.deepcopy(t);wrong['preservation_positions'].append(c['action_index'])
        with pytest.raises(ValueError):evaluator.validate_mask(c,wrong)
        wrong=copy.deepcopy(t);wrong['preservation_positions'].remove(len(t['action_ids'])-1)
        with pytest.raises(ValueError):evaluator.validate_mask(c,wrong)
        wrong=copy.deepcopy(c);wrong['action_index']+=1
        with pytest.raises(ValueError):evaluator.validate_mask(wrong,t)


def test_frozen384_no_missing_duplicate_or_missplit():
    p=frozen_packet();evaluator.validate_packet(p)
    for mutate in [lambda x:x['records'].pop(),lambda x:x['records'][0].update(example_id=x['records'][1]['example_id']),
                   lambda x:x['records'][0].update(split='support47'),lambda x:x['support_image_ids'].append('529411'),
                   lambda x:x['positive_cases'].pop()]:
        bad=copy.deepcopy(p);mutate(bad)
        with pytest.raises(ValueError):evaluator.validate_packet(bad)


def test_candidate_recipe_and_seven_state_binding_fail_before_load(tmp_path,monkeypatch):
    packet=frozen_packet();monkeypatch.setattr(evaluator,'TRAINING',tmp_path)
    receipt=dict(schema_version='selective_preservation_seven.training.v1',status='completed',numerical_admission=evaluator.ADMISSION,
        updates=81,stop_reason='fixed_steps',lambda_kl=10.,lambda_support_kl=100.,positive_image_ids=[c['image_id'] for c in packet['positive_cases']],
        support_image_ids=packet['support_image_ids'],local_KL_states=273,positive_trajectory_tokens=325,support_action_states=5838,total_cached_states=6111,
        adapter={'root':str(tmp_path/'adapter')},inputs_sha256='unused')
    for change in [dict(schema_version='selective_preservation_strong.training.v1'),dict(status='provisional'),dict(updates=23),dict(stop_reason='margin'),
        dict(local_KL_states=325),dict(positive_trajectory_tokens=273),dict(support_action_states=5848),dict(support_image_ids=packet['support_image_ids']+['529411']),
        dict(adapter={'root':str(tmp_path/'old/adapter')}),dict(positive_image_ids=['368','7116'])]:
        with pytest.raises(ValueError):evaluator.validate_receipt(dict(receipt,**change),packet)


def test_eight_complete384_shards_single_candidate_only():
    p={'packet':'frozen'};ts=[dict(shard=i,gpu=i,status='completed',manifest_sha256=evaluator.digest(p),continuations=48,model_loads=1,
        score_forwards=7 if i==0 else 0,new_tokens=100,elapsed_seconds=1.,training_receipt_sha256='same') for i in range(8)]
    evaluator.require_complete_shards(ts,p)
    for change in [dict(status='failed'),dict(continuations=47),dict(score_forwards=2),dict(training_receipt_sha256='other'),dict(new_tokens=148033)]:
        wrong=copy.deepcopy(ts);wrong[0].update(change)
        with pytest.raises(ValueError):evaluator.require_complete_shards(wrong,p)
    with pytest.raises(ValueError):evaluator.require_complete_shards(ts[:7],p)


def test_full_natural_prompt_remains_unforced():
    from probes.dora_owner_learning.tests.test_entrance_ce_eval import natural_fixture
    tok,frozen,row=natural_fixture();evaluator.validate_natural(row,frozen,tok)
    for change in [dict(prefix_ids=[1]),dict(forced_ids=[1]),dict(action_ids=[1]),dict(remaining_budget=3085)]:
        with pytest.raises(ValueError):evaluator.validate_natural(dict(row,**change),frozen,tok)
