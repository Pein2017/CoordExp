import copy
import pytest
import torch
from probes.dora_owner_learning import selective_preservation_seven as seven


@pytest.fixture(scope='module')
def data():return seven.build_data()


def test_real_seven_masks_objective_and_reference_conflict_removal(data):
    old,cases,trajectories,support,_=data
    assert [c['image_id'] for c in cases]==list(seven.MASKS)
    assert sum(len(trajectories[c['case_id']]['preservation_positions']) for c in cases)==273
    assert len(support)==47 and sum(len(c['action_ids']) for c in support)==5838
    assert [c['image_id'] for c in support]==[c['image_id'] for c in old['support_cases'] if c['image_id']!='529411']
    for c in old['cases']:assert trajectories[c['case_id']]==old['trajectories'][c['case_id']]
    result=seven.objective_fixture(cases,trajectories)
    assert result['status']=='passed' and len(result['real_token_masks'])==7 and result['max_gradient_weight_error']<1e-6


def test_objective_fixture_rejects_unscaled_old_half_weight(data,monkeypatch):
    _,cases,trajectories,_,_=data;real=seven.global_image_loss
    def wrong(*args):
        loss,ce,kl=real(*args)
        return (loss*3.5 if ce is not None else loss),ce,kl
    monkeypatch.setattr(seven,'global_image_loss',wrong)
    with pytest.raises(ValueError,match='mean7'):
        seven.objective_fixture(cases,trajectories)


def test_shifted_generic_target_and_excluded_nonentry_mask_fail(data):
    _,cases,trajectories,_,_=data;c=cases[2];t=trajectories[c['case_id']]
    item=dict(kind='entrance',case=c,action_ids=t['action_ids'],positions=t['preservation_positions'],suffix_start=t['suffix_start'])
    logits=torch.zeros(len(t['action_ids']),152670);targets=torch.tensor(t['action_ids']);ref=torch.log_softmax(logits[item['positions']],-1)
    for bad in [dict(item,case={**c,'target_token_id':c['target_token_id']+1}),dict(item,case={**c,'action_index':c['action_index']+1}),
                dict(item,positions=[c['action_index']]+item['positions'][1:])]:
        with pytest.raises(ValueError):seven.global_image_loss(logits,targets,bad,ref)


def test_eight_rank54_assignment_and_fixed_dose_counts(data):
    _,cases,trajectories,support,_=data;p=dict(cases=cases,trajectories=trajectories,support_cases=support)
    work=[seven.rank_items(p,r) for r in range(8)]
    assert [len(w) for w in work]==[7]*7+[5]
    assert len({x['key'] for w in work for x in w})==54
    assert [sum(x['kind']=='entrance' for x in w) for w in work]==[1]*7+[0]
    assert [len(w)*82+(14 if r==0 else 0) for r,w in enumerate(work)]==seven.CEILINGS
    assert sum(seven.CEILINGS)==4442 and 7*81==567 and 47*81==3807


def fixture(tmp_path,wrong_dose=False):
    seven.publish(tmp_path/'launcher_exit.json',dict(exit_code=0))
    for rank in range(8):
        path=tmp_path/'ranks'/f'rank{rank}'/'terminal.json';path.parent.mkdir(parents=True)
        seven.publish(path,dict(rank=rank,status='completed',updates=23 if wrong_dose else 81,model_forwards=seven.CEILINGS[rank],cumulative_model_seconds=1,
            counters=dict(support_KL_trajectories=486 if rank<7 else 405,supervised_target_tokens=81 if rank<7 else 0,
                          reference_forwards=7 if rank<7 else 5,train_forwards=567 if rank<7 else 405,entry_score_forwards=14 if rank==0 else 0),
            state=dict(adapter_hash='same',optimizer_hash='same',reference_cache_bytes=1,peak_cuda_allocated_bytes=1,peak_cuda_reserved_bytes=1,peak_rss_bytes=1)))
    root=tmp_path/'adapter';root.mkdir()
    for name in ['adapter_config.json','adapter_model.safetensors']:(root/name).write_bytes(b'fixture')
    identity=dict(root=str(root),files=[dict(relative_path=p.name,sha256=seven.file_hash(p)) for p in sorted(root.iterdir())])
    seven.publish(tmp_path/'proof.json',{})
    seven.publish(tmp_path/'fixture.json',dict(status='passed',old_half_weight_rejected=True,state_pooling_rejected=True,compensation_and_clip_checks=True))
    seven.publish(tmp_path/'smoke.json',dict(status='passed',updates=[1,2],initial_reference_count=54,initial_KL_zero=True,
        step2_max_KL_by_rank={'0':.1},reduced_gradient_adapter_optimizer_rank_identity=True,frozen_bytes_unchanged=True))
    ref=lambda name:dict(path=str(tmp_path/name),sha256=seven.file_hash(tmp_path/name))
    seven.publish(tmp_path/'provisional.json',dict(status='unsealed',updates=81,numerical_admission=seven.NUMERICAL_ADMISSION,cases=[{}]*7,
        support_image_ids=list(map(str,range(47))),adapter=identity,source_embedding=identity,
        prior_path_proof=ref('proof.json'),objective_fixture=ref('fixture.json'),two_step=ref('smoke.json')))


def test_sealing_requires81_steps_and_current_global_counts(tmp_path):
    fixture(tmp_path)
    r=seven.finalize(tmp_path)
    assert r['status']=='completed' and r['actual_positive_uses']==567 and r['actual_support_uses']==3807
    with pytest.raises(ValueError):seven.finalize(tmp_path)


def test_old23_step_terminal_cannot_seal_current_arm(tmp_path):
    fixture(tmp_path,wrong_dose=True)
    with pytest.raises(ValueError):seven.finalize(tmp_path)
    assert not (tmp_path/'receipt.json').exists()


def test_occupied_output(tmp_path):
    with pytest.raises(ValueError):seven.prepare(tmp_path)
