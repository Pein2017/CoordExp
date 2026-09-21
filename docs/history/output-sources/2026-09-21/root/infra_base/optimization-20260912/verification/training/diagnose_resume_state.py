"""Project authenticated rank-zero exact-state differences and optimizer restore."""
import importlib.util
import json
from collections import Counter
from dataclasses import fields
from pathlib import Path
import sys

ROOT = Path('/data/CoordExp/outputs/infra_base/optimization-20260912')
sys.path.insert(0, str(ROOT / 'scripts'))
import compare_resume as compare
import torch
from src.artifacts.training_state import _optimizer_load_state_dict


def admit(path):
    manifest = compare.load_training_state_manifest(path)
    expectations = compare.TrainingStateExpectations(
        checkpoint_step=manifest.checkpoint_step, world_size=8,
        identities=manifest.identities, scheduler_applicable=True,
        scaler_applicable=False,
    )
    return compare.admit_training_state(path, expectations, current_rank=0).decoded_rank


torch.set_num_threads(1)
left = admit(ROOT / 'runs/candidate/checkpoints/step-4')
right = admit(ROOT / 'runs/candidate-resume-02/checkpoints/step-4')
receipt = {'scope': 'authenticated rank zero', 'field_comparisons': {}, 'tensor_examples': []}
for field in fields(compare.DecodedRankTrainingState):
    c = compare.ExactComparison()
    c.compare(getattr(left, field.name), getattr(right, field.name), field.name)
    receipt['field_comparisons'][field.name] = dict(c.counts)
for name in list(left.trainable_model)[:4]:
    a, b = left.trainable_model[name], right.trainable_model[name]
    delta = (a.double()-b.double()).abs()
    receipt['tensor_examples'].append({'name': name, 'dtype': str(a.dtype), 'shape': list(a.shape), 'unequal_elements': int((a!=b).sum()), 'max_abs_diff': delta.max().item(), 'rms_diff': delta.square().mean().sqrt().item()})
del left, right
parent = admit(ROOT / 'runs/candidate/checkpoints/step-2')


class NamedParameters(torch.nn.Module):
    def __init__(self, named):
        super().__init__()
        self.names = list(named)
        self.values = torch.nn.ParameterList([torch.nn.Parameter(v.clone()) for v in named.values()])

    def named_parameters(self, *args, **kwargs):
        return iter(zip(self.names, self.values))


model = NamedParameters(parent.trainable_model)
by_name = dict(model.named_parameters())
groups = [dict(group['options'], params=[by_name[name] for name in group['params']]) for group in parent.optimizer['param_groups']]
optimizer = torch.optim.AdamW(groups)
optimizer.load_state_dict(_optimizer_load_state_dict(parent.optimizer, model=model, optimizer=optimizer))
restored = dict(parent.optimizer, param_groups=[], state={})
for group in optimizer.param_groups:
    original = parent.optimizer['param_groups'][len(restored['param_groups'])]
    restored['param_groups'].append({'params': original['params'], 'options': {k:v for k,v in group.items() if k!='params'}})
    for name in original['params']:
        restored['state'][name] = optimizer.state[by_name[name]]
c = compare.ExactComparison()
c.compare(parent.optimizer, restored, 'optimizer_cpu_restore')
receipt['optimizer_cpu_restore'] = {'counts': dict(c.counts), 'mismatches': c.mismatches, 'scope': 'Real step-two tensor shapes/values/options through repository mapping and Torch AdamW.load_state_dict; CPU only, excludes CUDA execution.'}
receipt['parent_optimizer_groups'] = [{k:v for k,v in g['options'].items() if k in ('lr','initial_lr','betas','eps','foreach','fused','capturable','differentiable','amsgrad','weight_decay')} for g in parent.optimizer['param_groups']]
receipt['parent_optimizer_steps'] = dict(Counter(float(v['step']) for v in parent.optimizer['state'].values()))
receipt['parent_trainable_dtypes'] = dict(Counter(str(v.dtype) for v in parent.trainable_model.values()))
out=ROOT/'verification/training/resume-state-diagnosis-v2.json'
out.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
print(json.dumps(receipt,sort_keys=True))
