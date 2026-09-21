import copy
import math

import pytest
import torch

from probes.dora_owner_learning.selective_preservation_dense import (
    CEILINGS, compensate_for_ddp, file_hash, finalize, global_image_loss, prepare, publish, rank_items,
)


def clipped(vector):
    return vector * min(1., 1. / float(vector.norm()))


def test_distributed_loss_fixture_has_world_weight_localmean_and_clip_teeth():
    # Unequal local counts7/7/6x6, two distinct CE gradients and six support gradients/rank.
    local=[];counts=[]
    for rank in range(8):
        contributions=[torch.tensor([(i+1)/100,(-1)**i*.2],dtype=torch.float64)*(10/48) for i in range(rank,48,8)]
        if rank==0:contributions.insert(0,torch.tensor([4.,-1.],dtype=torch.float64))
        if rank==1:contributions.insert(0,torch.tensor([-.5,3.],dtype=torch.float64))
        parameter=torch.nn.Parameter(torch.tensor([.2,.3],dtype=torch.float64))
        for contribution in contributions:
            compensate_for_ddp(parameter.dot(contribution)).backward()
        local.append(parameter.grad.clone());counts.append(len(contributions))
    expected=sum(v/8 for v in local)
    ddp_averaged=torch.stack(local).mean(0)
    assert counts==[7,7,6,6,6,6,6,6]
    assert torch.allclose(ddp_averaged,expected)
    assert not torch.allclose(torch.stack([v/8 for v in local]).mean(0),expected)  # omit world compensation
    assert not torch.allclose(torch.stack([v/n for v,n in zip(local,counts)]).mean(0),expected)  # rank-local mean
    assert not torch.allclose(torch.stack([clipped(v) for v in local]).mean(0),clipped(expected))  # local preclip
    # Direct unpartitioned sum is an independent reference, not the rank-computed expression.
    direct=torch.tensor([3.5,2.],dtype=torch.float64)
    direct+=sum(torch.tensor([(i+1)/100,(-1)**i*.2],dtype=torch.float64)*(10/48) for i in range(48))
    assert torch.allclose(ddp_averaged,direct)


def test_dense_support_full_state_weight_reference_gradient_and_eos():
    ids=[0,1,151645]
    logits=torch.full((3,151646),-1000.)
    logits[:,:2]=math.log(.5);logits.requires_grad_(True)
    ref=torch.full_like(logits,-1000.)
    ref[:,0]=math.log(.8);ref[:,1]=math.log(.2);ref.requires_grad_(True)
    item=dict(kind='support',positions=[0,1,2],action_ids=ids)
    loss,ce,kl=global_image_loss(logits,torch.tensor(ids),item,ref)
    assert ce is None
    assert float(loss.detach())==pytest.approx((10/48)*(.8*math.log(1.6)+.2*math.log(.4)),abs=1e-7)
    compensate_for_ddp(loss).backward()
    expected=torch.tensor([-.3,.3])*(8*10/(48*3))
    assert torch.allclose(logits.grad[-1,:2],expected,atol=1e-7)
    assert ref.grad is None
    with pytest.raises(ValueError):
        global_image_loss(logits,torch.tensor(ids),{**item,'positions':[0,1]},ref)


def test_rank_assignment_six_support_and_exact_entry_ownership():
    cases=[dict(case_id='a'),dict(case_id='b')]
    packet=dict(cases=cases,trajectories={c['case_id']:dict(action_ids=[1,151645],preservation_positions=[0],suffix_start=1) for c in cases},
                support_cases=[dict(example_id=f's{i}',action_ids=[1,151645]) for i in range(48)])
    work=[rank_items(packet,r) for r in range(8)]
    assert [len(x) for x in work]==[7,7,6,6,6,6,6,6]
    assert len({x['key'] for rank in work for x in rank})==50
    assert work[0][0]['key']=='a' and work[1][0]['key']=='b'
    assert [x['key'] for x in work[2]]==[f's{i}' for i in range(2,48,8)]


def terminal_fixture(tmp_path,*,exit_code=0,bad_rank=None):
    publish(tmp_path/'launcher_exit.json',dict(exit_code=exit_code))
    for rank in range(8):
        path=tmp_path/'ranks'/f'rank{rank}'/'terminal.json';path.parent.mkdir(parents=True)
        state=dict(adapter_hash='same',optimizer_hash='same',reference_cache_bytes=1,peak_cuda_allocated_bytes=1,
                   peak_cuda_reserved_bytes=1,peak_rss_bytes=1)
        publish(path,dict(rank=rank,status='failed' if bad_rank==rank else 'completed',updates=23,model_forwards=CEILINGS[rank],state=state,
            cumulative_model_seconds=1,counters=dict(supervised_target_tokens=23 if rank<2 else 0,reference_forwards=7 if rank<2 else 6,
                                                   train_forwards=161 if rank<2 else 138,entry_score_forwards=48 if rank==0 else 0)))
    adapter=tmp_path/'adapter';adapter.mkdir()
    for name in ['adapter_model.safetensors','adapter_config.json']:(adapter/name).write_bytes(b'fixture')
    identity=dict(root=str(adapter),files=[dict(relative_path=p.name,sha256=file_hash(p)) for p in sorted(adapter.iterdir())])
    publish(tmp_path/'provisional.json',dict(status='unsealed',adapter=identity,source_embedding=identity))


def test_completed_seal_requires_all_rank_terminals_and_launcher_exit0(tmp_path):
    terminal_fixture(tmp_path)
    receipt=finalize(tmp_path)
    assert receipt['status']=='completed' and receipt['global_model_forwards']==1248
    assert [r['rank'] for r in receipt['rank_terminals']]==list(range(8))
    with pytest.raises(ValueError):finalize(tmp_path)


@pytest.mark.parametrize('failure',['rank','exit','extra_rank'])
def test_unsealed_failure_cannot_be_promoted(tmp_path,failure):
    terminal_fixture(tmp_path,exit_code=1 if failure=='exit' else 0,bad_rank=3 if failure=='rank' else None)
    if failure=='extra_rank':(tmp_path/'ranks'/'rank8').mkdir()
    with pytest.raises(ValueError):finalize(tmp_path)
    assert not (tmp_path/'receipt.json').exists()


def test_occupied_output(tmp_path):
    with pytest.raises(ValueError):prepare(tmp_path)
