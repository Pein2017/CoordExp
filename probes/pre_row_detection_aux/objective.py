"""Differentiable final-norm opener rows and the frozen training-only head."""
from __future__ import annotations
import math
from pathlib import Path
import torch
import torch.nn.functional as F
from .bank import require
SEED = 92711
INITIALIZER = 'torch.nn.Linear.reset_parameters: kaiming_uniform_(a=sqrt(5)); bias uniform +/-1/sqrt(fan_in); isolated CPU generator seed92711'

class TrainableRows:
    """One final norm call, selected FP32 rows, original autograd connection."""
    def __init__(self, norm, positions):
        self.norm, self.positions = norm, list(positions)
        require(self.positions and all(type(p) is int and p >= 0 for p in self.positions), 'invalid norm positions')
        self.hidden = None

    def hook(self, module, args, output):
        require(self.hidden is None, 'repeated final norm call')
        require(isinstance(output, torch.Tensor) and output.ndim == 3 and output.shape[0] == 1 and max(self.positions) < output.shape[1], 'invalid norm output')
        self.hidden = output[0, self.positions].float()

    def __enter__(self):
        self.handle = self.norm.register_forward_hook(self.hook)
        return self

    def __exit__(self, kind, value, traceback):
        self.handle.remove()
        if kind is None:
            require(self.hidden is not None, 'missing final norm call')


def make_head(width, classes):
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(SEED)
        return torch.nn.Linear(width, classes + 4, dtype=torch.float32)


def boxes(logits):
    a, b, c, d = logits.float().sigmoid().unbind(-1)
    return torch.stack((a,b,a+(1-a)*c,b+(1-b)*d), -1)


def giou(pred, target):
    area = lambda b: (b[...,2:]-b[...,:2]).clamp_min(0).prod(-1)
    intersection = (torch.minimum(pred[...,2:],target[...,2:])-torch.maximum(pred[...,:2],target[...,:2])).clamp_min(0).prod(-1)
    union = area(pred)+area(target)-intersection
    enclosure = area(torch.cat((torch.minimum(pred[...,:2],target[...,:2]),torch.maximum(pred[...,2:],target[...,2:])), -1))
    eps = torch.finfo(torch.float32).eps
    return intersection/union.clamp_min(eps) - (enclosure-union)/enclosure.clamp_min(eps)


def auxiliary(head, hidden, rows, classes, denominator):
    require(hidden.requires_grad, 'detached auxiliary state')
    require(denominator > 0 and len(rows) == len(hidden), 'aux row normalization drift')
    output = head(hidden.float())
    targets = torch.tensor([classes.index(r['class']) for r in rows], device=output.device)
    gt = torch.tensor([r['box'] for r in rows], device=output.device, dtype=torch.float32)/1000
    pred = boxes(output[:,len(classes):])
    terms = dict(CE=F.cross_entropy(output[:,:len(classes)], targets, reduction='none')/math.log(max(len(classes),2)),
        L1=(pred-gt).abs().mean(-1), GIoU=.5*(1-giou(pred,gt)))
    weights = torch.tensor([r['weight'] for r in rows], device=output.device)/denominator
    reduced = {k:(v*weights).sum() for k,v in terms.items()}
    return sum(reduced.values()), dict(components={k:float(v.detach()) for k,v in reduced.items()},
        degenerate=int(((pred[:,2:]-pred[:,:2])==0).any(-1).sum()), rows=len(rows))


def clip_separately(backbone, head):
    return dict(backbone=float(torch.nn.utils.clip_grad_norm_(backbone,1,error_if_nonfinite=True)),
        head=float(torch.nn.utils.clip_grad_norm_(head,1,error_if_nonfinite=True)))


def save_head(head, path, bank_sha256, classes):
    require(not Path(path).exists(), 'head output exists')
    torch.save(dict(schema='pre-row-training-head-v1', bank_sha256=bank_sha256, classes=classes,
        initializer=INITIALIZER, state_dict={k:v.detach().cpu() for k,v in head.state_dict().items()}),path)


def load_head(head, path, bank_sha256, classes):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    require(payload['schema'] == 'pre-row-training-head-v1' and payload['bank_sha256'] == bank_sha256 and payload['classes'] == classes and payload['initializer'] == INITIALIZER, 'head identity drift')
    head.load_state_dict(payload['state_dict'], strict=True)

