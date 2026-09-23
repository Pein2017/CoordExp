import json
from pathlib import Path
from types import SimpleNamespace
import torch
from src.config.loader import load_train_config
from src.optim.factory import build_optimizer_and_scheduler
from probes.training_set_completion.coordinate_codebook_alignment.three_loss_prepare import ROOT, OLD, V3
from probes.training_set_completion.coordinate_codebook_alignment import scale_train as scale


def test_frozen_config_schedule_and_real_scheduler_prefix():
    new=load_train_config(ROOT/'training-config-v1.json').config
    old=load_train_config(OLD/'training-config-repair-v2.json').config
    assert new.losses.normalizer == 'segment_balanced'
    p=new.losses.protected
    assert (p.base_ce.weight,p.token_type_gate.weight,p.raw_axis_validity_hinge.weight,p.coord_gaussian_rps.weight)==(1,.2,.01,0)
    assert p.raw_axis_validity_hinge.margin == 1/999
    assert tuple(p.token_type_gate.groups)==('desc_text','schema','coordinate','eos')
    assert new.optimizer == old.optimizer
    sequences=[]
    for cfg,horizon in [(old,1968),(new,984)]:
        parameter=torch.nn.Parameter(torch.tensor(1.))
        groups=[{'params':[parameter],'lr':.001,'weight_decay':0.}]
        plan=SimpleNamespace(groups=groups,to_torch_param_groups=lambda:groups)
        optimizer,scheduler=build_optimizer_and_scheduler(cfg.optimizer,plan,total_training_steps=horizon)
        lrs=[]
        for step in range(984):
            lrs.append(optimizer.param_groups[0]['lr'])
            parameter.grad=torch.ones_like(parameter);optimizer.step();scheduler.step()
        sequences.append(lrs)
    assert sequences[0]==sequences[1] and sequences[1][0]==0 and sequences[1][1]>0
    packet=json.loads((ROOT/'packing-exposure-v1.json').read_text())
    prior=json.loads((V3/'packing-exposure.json').read_text())
    assert packet['global_pack_indices']==prior['global_pack_indices'][:7872]
    original=scale.EXPECTED_UPDATES
    try:
        scale.EXPECTED_UPDATES=984
        for rank in range(4): assert len(scale.expected_first_two_updates(packet,rank=rank))==4
    finally:scale.EXPECTED_UPDATES=original
