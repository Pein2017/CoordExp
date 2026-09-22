import copy
import pytest
from probes.dora_owner_learning import selective_preservation_stable as stable


@pytest.fixture(scope='module')
def data():return stable.build_data()


def test_exact_three_Source_EOS_references_and_current_objective(data):
    old,support,added,cards,_=data
    assert len(support)==50 and sum(len(c['action_ids']) for c in support)==6024
    assert [c['image_id'] for c in added]==['73843','360573','545632']
    assert [(len(c['action_ids']),len(c['action_ids'])+len(c['prompt_token_ids'])) for c in added]==[(10,1372),(95,1415),(81,1365)]
    for c in old['support_cases']:assert c==next(x for x in support if x['image_id']==c['image_id'])
    result=stable.objective_fixture(old,support,added,cards)
    assert result['status']=='passed' and result['positive_and_local_unchanged'] and result['support_denominator']==50
    assert result['max_gradient_weight_error']<1e-6


def test_old47_denominator_is_rejected(data,monkeypatch):
    old,support,added,cards,_=data
    monkeypatch.setattr(stable,'global_image_loss',stable.seven47_loss)
    with pytest.raises(ValueError,match='support50 rather than47'):
        stable.objective_fixture(old,support,added,cards)


def test_missing_modified_or_non_Source_reference_rejected(data):
    _,_,added,cards,_=data
    with pytest.raises(ValueError):stable.validate_new_references(added[:-1],cards)
    for field in ['token','EOS','prompt']:
        altered=copy.deepcopy(added)
        if field=='token':altered[0]['action_ids'][1]+=1
        elif field=='EOS':altered[0]['action_ids'][-1]=1
        else:altered[0]['prompt_token_ids'][0]+=1
        with pytest.raises(ValueError):stable.validate_new_references(altered,cards)


def test_current_rank_items_counts_not_previous47(data):
    old,support,_,_,_=data;p=dict(cases=old['cases'],trajectories=old['trajectories'],support_cases=support)
    work=[stable.rank_items(p,r) for r in range(8)]
    assert [len(w) for w in work]==[8,8,7,7,7,7,7,6]
    assert len({x['key'] for w in work for x in w})==57
    assert [len(w)*82+(14 if r==0 else 0) for r,w in enumerate(work)]==stable.CEILINGS
    assert sum(stable.CEILINGS)==4688 and 50*81==4050


def fixture(tmp_path,old_counter=False):
    stable.publish(tmp_path/'launcher_exit.json',dict(exit_code=0))
    for rank in range(8):
        path=tmp_path/'ranks'/f'rank{rank}'/'terminal.json';path.parent.mkdir(parents=True)
        stable.publish(path,dict(rank=rank,status='completed',updates=81,model_forwards=stable.CEILINGS[rank],cumulative_model_seconds=1,
            counters=dict(support_KL_trajectories=405 if old_counter else (567 if rank<2 else 486),supervised_target_tokens=81 if rank<7 else 0,
                          reference_forwards=[8,8,7,7,7,7,7,6][rank],train_forwards=[8,8,7,7,7,7,7,6][rank]*81,entry_score_forwards=14 if rank==0 else 0),
            state=dict(adapter_hash='same',optimizer_hash='same',reference_cache_bytes=1,peak_cuda_allocated_bytes=1,peak_cuda_reserved_bytes=1,peak_rss_bytes=1)))
    root=tmp_path/'adapter';root.mkdir()
    for n in ['adapter_config.json','adapter_model.safetensors']:(root/n).write_bytes(b'fixture')
    identity=dict(root=str(root),files=[dict(relative_path=p.name,sha256=stable.file_hash(p)) for p in sorted(root.iterdir())])
    stable.publish(tmp_path/'proof.json',{})
    stable.publish(tmp_path/'fixture.json',dict(status='passed',support_denominator=50,positive_and_local_unchanged=True,
        old47_denominator_rejected=True,new_Source_EOS_references_verified=True,compensation_and_clip_checks=True))
    stable.publish(tmp_path/'smoke.json',dict(status='passed',updates=[1,2],initial_reference_count=57,initial_KL_zero=True,
        step2_max_KL_by_rank={'0':.1},reduced_gradient_adapter_optimizer_rank_identity=True,frozen_bytes_unchanged=True))
    ref=lambda name:dict(path=str(tmp_path/name),sha256=stable.file_hash(tmp_path/name))
    stable.publish(tmp_path/'provisional.json',dict(status='unsealed',updates=81,numerical_admission=stable.NUMERICAL_ADMISSION,
        cases=[{}]*7,support_image_ids=list(map(str,range(50))),adapter=identity,source_embedding=identity,
        prior_path_proof=ref('proof.json'),objective_fixture=ref('fixture.json'),two_step=ref('smoke.json')))


def test_complete_current81_global_counts_seal(tmp_path):
    fixture(tmp_path);r=stable.finalize(tmp_path)
    assert r['global_model_forwards']==4688 and r['actual_positive_uses']==567 and r['actual_support_uses']==4050
    with pytest.raises(ValueError):stable.finalize(tmp_path)


def test_previous_reference_use_count_cannot_seal(tmp_path):
    fixture(tmp_path,old_counter=True)
    with pytest.raises(ValueError):stable.finalize(tmp_path)
    assert not (tmp_path/'receipt.json').exists()


def test_occupied_output(tmp_path):
    with pytest.raises(ValueError):stable.prepare(tmp_path)
