"""Exact incremental provenance replay for the frozen coordinate-margin states.

The source prefix is materialised with the accepted native batch builder.  The
saved continuation is then consumed through the model's ordinary incremental
``generate`` path, with a processor that forces only the recorded tokens before
the requested score.  The processor returns the unmodified score at the target
step; it never patches a model output or a hidden state.

This producer deliberately keeps reduction out of the runtime.  It writes one
``capture.pt`` per state with the tensors consumed by
``coordinate_margin/reduce.py`` and a small JSON receipt carrying all source
bindings and replay parity checks.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch
from transformers import (
    GenerationConfig,
    LogitsProcessor,
    LogitsProcessorList,
    StoppingCriteria,
    StoppingCriteriaList,
)

from probes.training_set_completion.numerical_feedback.runtime import full_prefix
from probes.training_set_completion.readout_norm_fresh import _binding, _input_identity
from probes.training_set_completion.untied_shared import load_model
from src.inference.bound_requests import build_bound_native_requests
from src.qwen.native import prepare_native_inputs


EOS = 151645
COORD = 151670
VOCAB_PREFIX = 4
PARITY_ATOL = 2e-4


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def token_hash(tokens: list[int]) -> str:
    return canonical_hash(tokens)


def tensor_hash(value: torch.Tensor) -> str:
    data = value.detach().cpu().contiguous()
    return hashlib.sha256(data.view(torch.uint8).numpy().tobytes()).hexdigest()


def _path_binding(path: Path) -> dict[str, Any]:
    return _binding(path)


def _finite(value: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise RuntimeError(f"nonfinite tensor: {name}")


def _target_row(value: Any, target_index: int) -> torch.Tensor:
    if isinstance(value, (tuple, list)):
        value = value[0]
    if not isinstance(value, torch.Tensor) or value.ndim < 2:
        raise TypeError("hook output is not a tensor with a sequence dimension")
    if target_index >= value.shape[0]:
        raise IndexError("target index is outside hook batch")
    return value[target_index, -1]


def _source_sha(path: Path, expected: dict[str, Any] | None) -> dict[str, Any]:
    observed = _path_binding(path)
    if expected is not None and observed.get("sha256") != expected.get("sha256"):
        raise ValueError(f"source binding changed: {path}")
    return observed


def _state_source(state: dict[str, Any]) -> tuple[Path, Path, Path]:
    return (
        Path(state["source_release"]["path"]),
        Path(state["source_trajectory"]["path"]),
        Path(state["effective_readout"]["path"]),
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _continuation_from_companion(
    row: dict[str, Any], source_end: int, count: int, pad: int
) -> list[int]:
    """Reproduce the source generator's suffix for a non-target companion."""
    tokens = [int(x) for x in row["token_ids"]]
    suffix: list[int] = []
    ended = False
    for index in range(count):
        absolute = source_end + index
        if ended or absolute >= len(tokens):
            suffix.append(pad)
            continue
        value = tokens[absolute]
        suffix.append(value)
        if value == EOS:
            ended = True
    return suffix


def _module_value(output: Any) -> torch.Tensor:
    if isinstance(output, (tuple, list)):
        output = output[0]
    if not isinstance(output, torch.Tensor):
        raise TypeError("module hook output is not a tensor")
    return output


class _ReplayCapture:
    """Hooks and score capture for exactly one target generation step."""

    def __init__(
        self,
        runtime: "MarginRuntime",
        target_index: int,
        target_step: int,
        instrument: bool,
    ) -> None:
        self.runtime = runtime
        self.target_index = target_index
        self.target_step = target_step
        self.instrument = instrument
        self.forward_count = 0
        self.active_step = -1
        self.last_head_input: torch.Tensor | None = None
        self.last_head_logits: torch.Tensor | None = None
        self.target_head_input: torch.Tensor | None = None
        self.target_logits: torch.Tensor | None = None
        self.target_logit_hook: torch.Tensor | None = None
        self.layers: dict[str, list[torch.Tensor | None]] = {
            "layer_inputs": [],
            "attention": [],
            "post_attention": [],
            "mlp": [],
            "layer_outputs": [],
        }
        self.pre_final: torch.Tensor | None = None
        self.handles: list[Any] = []

    def _clone(self, value: torch.Tensor) -> torch.Tensor:
        # The clone happens before the text model applies any DeepStack
        # post-layer in-place update to its caller-visible hidden state.
        return value[self.target_index, -1].detach().clone().cpu()

    def model_pre(self, _module: Any, _args: tuple[Any, ...], _kwargs: dict[str, Any]) -> None:
        self.active_step = self.forward_count
        self.forward_count += 1
        self.runtime.forward_count += 1
        if self.runtime.forward_count > self.runtime.args.max_forwards:
            raise RuntimeError("forward budget exhausted")
        if time.monotonic() - self.runtime.started > self.runtime.args.max_seconds:
            raise RuntimeError("wall-time budget exhausted")
        self.last_head_input = None
        self.last_head_logits = None

    def head_pre(self, _module: Any, inputs: tuple[Any, ...]) -> None:
        if not inputs:
            raise RuntimeError("lm head prehook received no input")
        value = inputs[0]
        if not isinstance(value, torch.Tensor):
            raise TypeError("lm head input is not a tensor")
        self.last_head_input = self._clone(value)
        if self.active_step == self.target_step:
            self.target_head_input = self.last_head_input.clone()

    def head_post(self, _module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
        value = _module_value(output)
        self.last_head_logits = self._clone(value)
        if self.active_step == self.target_step:
            self.target_logit_hook = self.last_head_logits.clone()

    def layer_pre(self, index: int, _module: Any, inputs: tuple[Any, ...]) -> None:
        if self.active_step != self.target_step or not inputs:
            return
        value = inputs[0]
        if not isinstance(value, torch.Tensor):
            raise TypeError("decoder layer input is not a tensor")
        self.layers["layer_inputs"][index] = self._clone(value)

    def attention_post(self, index: int, _module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
        if self.active_step != self.target_step:
            return
        value = _module_value(output)
        self.layers["attention"][index] = self._clone(value)

    def post_attention_pre(self, index: int, _module: Any, inputs: tuple[Any, ...]) -> None:
        if self.active_step != self.target_step or not inputs:
            return
        value = inputs[0]
        if not isinstance(value, torch.Tensor):
            raise TypeError("post-attention residual is not a tensor")
        self.layers["post_attention"][index] = self._clone(value)

    def mlp_post(self, index: int, _module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
        if self.active_step != self.target_step:
            return
        value = _module_value(output)
        self.layers["mlp"][index] = self._clone(value)

    def layer_post(self, index: int, _module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
        if self.active_step != self.target_step:
            return
        value = _module_value(output)
        self.layers["layer_outputs"][index] = self._clone(value)

    def norm_pre(self, _module: Any, inputs: tuple[Any, ...]) -> None:
        if self.active_step != self.target_step or not inputs:
            return
        value = inputs[0]
        if not isinstance(value, torch.Tensor):
            raise TypeError("final norm input is not a tensor")
        self.pre_final = self._clone(value)

    def install(self) -> None:
        model = self.runtime.model
        self.handles.append(model.register_forward_pre_hook(self.model_pre, with_kwargs=True))
        self.handles.append(self.runtime.head.register_forward_pre_hook(self.head_pre))
        self.handles.append(self.runtime.head.register_forward_hook(self.head_post))
        if not self.instrument:
            return
        for index, layer in enumerate(self.runtime.layers):
            self.layers["layer_inputs"].append(None)
            self.layers["attention"].append(None)
            self.layers["post_attention"].append(None)
            self.layers["mlp"].append(None)
            self.layers["layer_outputs"].append(None)
            self.handles.append(
                layer.register_forward_pre_hook(
                    lambda module, inputs, i=index: self.layer_pre(i, module, inputs)
                )
            )
            self.handles.append(
                layer.self_attn.register_forward_hook(
                    lambda module, inputs, output, i=index: self.attention_post(i, module, inputs, output)
                )
            )
            self.handles.append(
                layer.post_attention_layernorm.register_forward_pre_hook(
                    lambda module, inputs, i=index: self.post_attention_pre(i, module, inputs)
                )
            )
            self.handles.append(
                layer.mlp.register_forward_hook(
                    lambda module, inputs, output, i=index: self.mlp_post(i, module, inputs, output)
                )
            )
            self.handles.append(
                layer.register_forward_hook(
                    lambda module, inputs, output, i=index: self.layer_post(i, module, inputs, output)
                )
            )
        self.handles.append(self.runtime.final_norm.register_forward_pre_hook(self.norm_pre))

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def validate(self) -> None:
        if self.target_head_input is None or self.target_logit_hook is None:
            raise RuntimeError("target head capture was not observed")
        if not self.instrument:
            return
        for name, values in self.layers.items():
            if len(values) != len(self.runtime.layers) or any(value is None for value in values):
                raise RuntimeError(f"missing target layer capture: {name}")
        if self.pre_final is None:
            raise RuntimeError("missing target pre-final residual")


from probes.training_set_completion.coordinate_pair import runtime as qualified

class Captured(Exception):
    pass

class MarginRuntime:
    def __init__(self,args):
        self.args=args;self.started=time.monotonic();self.forward_count=0
        states=_read_json(args.selected_states)['states']
        self.state=next(x for x in states if x['id']==args.state_id)
        for k in ('source_release','source_trajectory','effective_readout'):
            _source_sha(Path(self.state[k]['path']),self.state[k])
        self.source=_read_json(Path(self.state['source_release']['path']))
        oldroot=Path(self.state['source_release']['path']).parents
        plan=Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-coordinate-pair-readout/execution-plan.json')
        panel=Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-numerical-recurrence-feedback/panel.json')
        opts=argparse.Namespace(plan=plan,panel=panel,output=args.output/args.state_id/'native',device=args.device,cell_ids=[self.source['job_id']],max_forwards=10000,max_seconds=7200,max_bytes=4*1024**3)
        self.native=qualified.Runtime(opts)
        self.model=self.native.model;self.head=self.native.head
        self.layers=list(self.model.model.language_model.layers)
        self.final_norm=self.model.model.language_model.norm
        self.target_index=int(self.source['batch_index']);self.offset=int(self.state['offset'])
        self.boundary=self.native.jobs[0]['boundary'];self.policy=self.state['source_policy']
        self.group,self.raw,_,_,self.batch=self.native._source(self.boundary)
        self.source_end=int(self.boundary['source_row']['end'])
        self.prefix=list(self.state['actual_prefix_token_ids'])
        assert self.prefix==self.source['target']['token_ids'][:self.offset]
        self.tsource=torch.load(self.state['source_trajectory']['path'],map_location='cpu',weights_only=False)
        self.cell=args.output/args.state_id

    def replay(self,instrument):
        inputs=qualified.component.full_prefix(self.batch,self.raw,self.source_end,int(self.native.q.tokenizer.pad_token_id),self.args.device)
        width=inputs['input_ids'].shape[1]
        assert inputs['input_ids'][self.target_index,-self.source_end:].tolist()==self.boundary['native_tokens'][:self.source_end]
        capture=_ReplayCapture(self,self.target_index,self.offset,instrument)
        runtime=self
        class Observe(LogitsProcessor):
            def __call__(self,ids,scores):
                step=ids.shape[1]-width
                assert ids[runtime.target_index,width:].tolist()==runtime.prefix[:step]
                if step==runtime.offset:
                    capture.target_logits=scores[runtime.target_index].detach().clone().cpu()
                    raise Captured()
                used,_,_=runtime.native._pair_scores(scores)
                assert int(used[runtime.policy][runtime.target_index].argmax())==runtime.prefix[step]
                return used[runtime.policy]
        capture.install()
        try:
            self.native.generate(inputs,processors=[Observe()],stopping=[])
            raise RuntimeError('capture not reached')
        except Captured:pass
        finally:capture.remove()
        capture.validate()
        return capture

    def run(self):
        no=self.replay(False);yes=self.replay(True)
        assert torch.equal(no.target_logits.view(torch.int32),yes.target_logits.view(torch.int32)), 'hook scores changed'
        assert torch.equal(no.target_head_input,yes.target_head_input)
        off=self.offset;sourcehead=self.tsource['head_inputs'][off];sourcecoord=self.tsource['raw_coordinate_logits'][off]
        logits=yes.target_logits;ids=self.native.ids.cpu()
        parity={'head_max_abs':float((yes.target_head_input-sourcehead).abs().max()),'coordinate_max_abs':float((logits[ids]-sourcecoord).abs().max()),'winner':int(logits.argmax()),'source_winner':self.state['original_winner'],'bitwise_hook_nohook':True}
        assert parity['head_max_abs']<=PARITY_ATOL and parity['coordinate_max_abs']<=PARITY_ATOL and parity['winner']==parity['source_winner'],parity
        x=yes.pre_final.to(self.args.device).float();scale=torch.rsqrt(x.pow(2).mean()+self.final_norm.variance_epsilon)
        assert getattr(self.head.base,'bias',None) is None
        t={k:torch.stack(v) for k,v in yes.layers.items()}
        t.update(input_residual=t['layer_inputs'][0],pre_final=yes.pre_final,norm_scale=scale.cpu(),norm_weight=self.final_norm.weight.detach().cpu(),norm_eps=self.final_norm.variance_epsilon,head_input=yes.target_head_input,logits=logits,no_hook_logits=no.target_logits,no_hook_head_input=no.target_head_input,coordinate_ids=ids,effective_W=self.native.U.cpu(),source_head=sourcehead,source_coordinate_logits=sourcecoord,state=self.state,parity=parity)
        torch.save(t,self.cell/'capture.pt')
        readout=torch.load(self.state['effective_readout']['path'],map_location='cpu',weights_only=False)
        assert torch.equal(t['effective_W'],readout['output_rows'])
        receipt={'status':'candidate_complete','state_id':self.state['id'],'pid':os.getpid(),'model_forwards':self.forward_count,'prefix_replays':2,'gpu_seconds':time.monotonic()-self.started,'tensor_bytes':(self.cell/'capture.pt').stat().st_size,'parity':parity,'capture':_path_binding(self.cell/'capture.pt'),'producer':_path_binding(Path(__file__)),'source_states':_path_binding(self.args.selected_states),'native_identity':self.native.ledger['identity'],'source_prefix_hash':qualified._token_hash(self.boundary['native_tokens'][:self.source_end]),'continuation_prefix_hash':qualified._token_hash(self.prefix),'input_identity':_input_identity(self.batch),'target_index':self.target_index,'batch_size':len(self.raw)}
        write_json(self.cell/'receipt.json',receipt)
        self.native.ledger['status']='forward_capture_complete';self.native.persist()
        print(json.dumps({'state':self.state['id'],'parity':parity,'forwards':self.forward_count}))

def main():
    p=argparse.ArgumentParser();p.add_argument('--selected-states',type=Path,required=True);p.add_argument('--state-id',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda');p.add_argument('--max-forwards',type=int,default=10000);p.add_argument('--max-seconds',type=float,default=7200);args=p.parse_args()
    runtime=None
    try:
        runtime=MarginRuntime(args);runtime.run()
    except BaseException as exc:
        write_json(args.output/args.state_id/'failure.json',{'error':repr(exc),'pid':os.getpid(),'model_forwards':runtime.forward_count if runtime else 0});raise

if __name__=='__main__':main()
