import copy
import inspect
import pytest

from probes.dora_owner_learning import selective_preservation_strong as strong


def test_coefficient_fixture_proves_support10x_old_two_unchanged_and_ddp_scaling():
    r=strong.coefficient_fixture()
    assert r['status']=='passed' and r['support_gradient_ratio']==10
    assert r['gradient_max_abs_error']<1e-15 and r['old_two_unchanged'] and r['compensation_and_clip_checks']


def test_coefficient_fixture_rejects_unchanged_support_weight(monkeypatch):
    monkeypatch.setattr(strong,'global_image_loss',strong.coefficient10_loss)
    with pytest.raises(ValueError,match='support coefficient'):
        strong.coefficient_fixture()


def fixture(tmp_path,*,support_uses=138,initial_zero=True,exit_code=0):
    strong.publish(tmp_path/'launcher_exit.json',dict(exit_code=exit_code))
    for rank in range(8):
        path=tmp_path/'ranks'/f'rank{rank}'/'terminal.json';path.parent.mkdir(parents=True)
        strong.publish(path,dict(rank=rank,status='completed',updates=23,model_forwards=strong.CEILINGS[rank],cumulative_model_seconds=1,
            counters=dict(support_KL_trajectories=support_uses,reference_forwards=7 if rank<2 else 6,
                          train_forwards=161 if rank<2 else 138,entry_score_forwards=48 if rank==0 else 0),
            state=dict(adapter_hash='same',optimizer_hash='same',reference_cache_bytes=1,peak_cuda_allocated_bytes=1,
                       peak_cuda_reserved_bytes=1,peak_rss_bytes=1)))
    root=tmp_path/'adapter';root.mkdir()
    for name in ('adapter_config.json','adapter_model.safetensors'):(root/name).write_bytes(b'fixture')
    identity=dict(root=str(root),files=[dict(relative_path=p.name,sha256=strong.file_hash(p)) for p in sorted(root.iterdir())])
    strong.publish(tmp_path/'proof.json',dict(prior_path=True))
    strong.publish(tmp_path/'fixture.json',dict(status='passed',support_gradient_ratio=10,old_two_unchanged=True,compensation_and_clip_checks=True))
    strong.publish(tmp_path/'smoke.json',dict(status='passed',updates=[1,2],initial_KL_zero=initial_zero,initial_reference_count=50,
        step2_max_KL_by_rank={'0':.01},reduced_gradient_adapter_optimizer_rank_identity=True,frozen_bytes_unchanged=True))
    ref=lambda name:dict(path=str(tmp_path/name),sha256=strong.file_hash(tmp_path/name))
    strong.publish(tmp_path/'provisional.json',dict(status='unsealed',numerical_admission=strong.NUMERICAL_ADMISSION,
        lambda_kl=10,lambda_support_kl=100,adapter=identity,source_embedding=identity,
        prior_path_proof=ref('proof.json'),coefficient_fixture=ref('fixture.json'),two_step=ref('smoke.json')))


def test_seal_requires_current_support1104_and_allrank_launcher_barrier(tmp_path):
    fixture(tmp_path)
    r=strong.finalize(tmp_path)
    assert r['status']=='completed' and r['actual_support_uses']==1104 and r['global_model_forwards']==1248
    with pytest.raises(ValueError):strong.finalize(tmp_path)


@pytest.mark.parametrize('failure',['stale_counter','initial_KL','launcher'])
def test_prior_proof_alone_cannot_seal_failed_current_arm(tmp_path,failure):
    fixture(tmp_path,support_uses=0 if failure=='stale_counter' else 138,
            initial_zero=failure!='initial_KL',exit_code=1 if failure=='launcher' else 0)
    with pytest.raises(ValueError):strong.finalize(tmp_path)
    assert not (tmp_path/'receipt.json').exists()


def test_current_execution_has_no_cross_coefficient_equivalence_gate():
    source=inspect.getsource(strong.execute_rank)
    assert 'vector_admission(' not in source and 'gradient_relative_l2(' not in source and 'SERIAL_CONTROL' not in source
    assert "counters['support_KL_trajectories']+=int(item['kind']=='support')" in source


def test_occupied_output(tmp_path):
    with pytest.raises(ValueError):strong.prepare(tmp_path)
