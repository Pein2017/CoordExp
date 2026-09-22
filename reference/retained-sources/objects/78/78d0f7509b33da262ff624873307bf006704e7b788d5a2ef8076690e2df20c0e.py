"""Single-image frozen native execution; request-scoped primary visual bank."""
from __future__ import annotations

import json
import fcntl
import os
from pathlib import Path
import sys
import time
import torch
from transformers import LogitsProcessor

from probes.training_set_completion.artifacts import binding
from src.artifacts.source_provenance import preserve_source
from src.qwen.native import exact_history_inputs

ROOT = Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-address-readout-pilot')


def write_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    return binding(path)


class RunLedger:
    """One GPU per job; package wall bound also implies the 8-GPU allocation bound."""
    def __init__(self, output_dir, device):
        self.output = Path(output_dir).resolve()
        if not self.output.is_relative_to(ROOT.resolve()):
            raise ValueError('all pilot writes must stay under the pilot output root')
        self.output.mkdir(parents=True, exist_ok=False)
        self.device = str(device)
        if self.device not in {f'cuda:{i}' for i in range(8)}:
            raise ValueError('pilot uses exactly one of cuda:0..7 per job')
        ROOT.mkdir(parents=True, exist_ok=True)
        self.gpu_lock = (ROOT / f'gpu-{self.device.split(":")[-1]}.lock').open('a')
        fcntl.flock(self.gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.started = time.time()
        self.counts = dict(model_forwards=0, vision_forwards=0)
        self.handles = []
        write_once(self.output / 'process.json', dict(pid=os.getpid(), device=self.device,
                   allocated_gpus=1, started_epoch=self.started, argv=sys.argv))

    def capture_sources(self):
        cwd = Path.cwd().resolve()
        sources, synthetic = set(), []
        for name,module in tuple(sys.modules.items()):
            filename = getattr(module,'__file__',None)
            if not isinstance(filename,str):
                continue
            path = Path(filename).resolve()
            if path.is_relative_to(cwd) and path.suffix == '.py':
                if path.is_file():
                    sources.add(path)
                elif name.startswith(('src.','probes.')):
                    raise FileNotFoundError(f'maintained source missing: {name}: {path}')
                else:
                    synthetic.append(dict(module=name,file_reference=filename))
        entries = []
        for path in sorted(sources):
            captured = preserve_source(path, run_root=self.output, relative_name=path.relative_to(cwd))
            entries.append(dict(source=binding(path), capture=binding(captured)))
        return write_once(self.output / 'sources.json', dict(entries=entries,non_file_external_module_references=synthetic))

    def attach(self, model):
        def count(_module, _args):
            wall = ROOT / 'wall-start.json'
            if not wall.exists():
                try:
                    write_once(wall, dict(started_epoch=time.time(), limit_seconds=14400, allocated_gpu_seconds_limit=115200))
                except FileExistsError:
                    pass
            start = json.loads(wall.read_text())['started_epoch']
            # Leave a shutdown margin before either package ceiling.
            if time.time() - start > 14340:
                raise RuntimeError('package wall-time budget exhausted')
            allocated = 0.0
            now = time.time()
            for process_path in ROOT.rglob('process.json'):
                process = json.loads(process_path.read_text())
                cost_path = process_path.with_name('cost.json')
                terminal = json.loads(cost_path.read_text())['terminal_epoch'] if cost_path.exists() else now
                allocated += (terminal-process['started_epoch']) * process['allocated_gpus']
            if allocated >= 115140:
                raise RuntimeError('package allocated GPU budget exhausted')
            if self.counts['model_forwards'] >= 600 or now-self.started >= 2400:
                raise RuntimeError('bounded qualification per-process budget exhausted')
            self.counts['model_forwards'] += 1
        self.handles.append(model.register_forward_pre_hook(count))
        self.handles.append(model.model.visual.register_forward_pre_hook(
            lambda *_: self.counts.__setitem__('vision_forwards', self.counts['vision_forwards'] + 1)))

    def finish(self, status, error=None):
        for handle in self.handles:
            handle.remove()
        terminal = time.time()
        receipt = write_once(self.output / 'cost.json', dict(status=status, error=error,
            started_epoch=self.started, terminal_epoch=terminal,
            allocated_gpu_seconds=terminal-self.started, **self.counts,
            artifact_bytes=sum(p.stat().st_size for p in self.output.rglob('*') if p.is_file())))
        self.gpu_lock.close()
        return receipt


class NativeCapture:
    def __init__(self, model, grid):
        if list(grid)[0] != 1 or int(grid[1]) % 2 or int(grid[2]) % 2:
            raise ValueError('requires one still-image grid with merge size two')
        self.model, self.grid = model, tuple(int(v) for v in grid)
        self.hidden = self.visual = None
        self.handles = []

    def __enter__(self):
        def head(_module, args):
            self.hidden = args[0].detach().clone()
        def visual(_module, _args, out):
            if out.ndim != 2 or out.shape != (self.grid[1]*self.grid[2]//4, 2048):
                raise ValueError('primary visual bank/image boundary mismatch')
            self.visual = out.detach().clone()
        self.handles = [self.model.get_output_embeddings().register_forward_pre_hook(head),
                        self.model.model.visual.merger.register_forward_hook(visual)]
        return self

    def __exit__(self, *_):
        for handle in self.handles:
            handle.remove()


def replay(model, batch, continuation):
    if len(batch.request_ids) != 1:
        raise ValueError('pilot caller requires one image per model call')
    prompt = list(batch.prompt_token_ids[0])
    history = prompt + list(continuation)
    inputs = exact_history_inputs(model, batch.inputs, [history], pad_token_id=0,
                                  logits_to_keep=len(continuation)+1)
    with NativeCapture(model, batch.image_grids[0]) as capture, torch.no_grad():
        logits = model(**inputs).logits[0]
    assert capture.hidden is not None and capture.visual is not None
    return logits, capture.hidden[0], capture.visual


class ReadoutProcessor(LogitsProcessor):
    def __init__(self, capture, bridge, parser, prompt_width):
        self.capture, self.bridge, self.parser = capture, bridge, parser
        self.prompt_width = prompt_width
        self.trace = []
        self.saved_logits = {}
        self.saved_roles = set()
        self.capture_logits = False

    def __call__(self, input_ids, scores):
        if input_ids.shape[0] != 1:
            raise ValueError('request-scoped bridge requires batch size one')
        prefix = input_ids[0, self.prompt_width:].tolist()
        role = self.parser(prefix)
        changed = scores
        if self.bridge is not None:
            if self.capture.hidden is None or self.capture.visual is None:
                raise ValueError('missing current request captures')
            changed = self.bridge(scores, self.capture.hidden[:, -1, :], self.capture.visual,
                                  torch.tensor([role], device=scores.device),
                                  self.capture.grid[1]//2, self.capture.grid[2]//2)
        if self.capture_logits and (not self.trace or role >= 0 and role not in self.saved_roles):
            self.saved_logits[len(self.trace)] = changed.detach().cpu().clone()
            self.saved_roles.add(role)
        chosen = int(changed[0].argmax())
        self.trace.append(dict(step=len(self.trace), token_id=chosen, role=role,
                               logprob=float(changed[0, chosen] - torch.logsumexp(changed[0], 0))))
        return changed


def generate(model, batch, tokenizer, bridge, parser, *, extension=(), max_new_tokens=64, capture_logits=False):
    from transformers import LogitsProcessorList
    if len(batch.request_ids) != 1:
        raise ValueError('single-image qualification caller')
    prompt = list(batch.prompt_token_ids[0])
    history = prompt + list(extension)
    inputs = {k:v for k,v in batch.inputs.items() if k not in ('input_ids','attention_mask','position_ids')}
    ids = torch.tensor([history], device=batch.inputs['input_ids'].device)
    inputs.update(input_ids=ids, attention_mask=torch.ones_like(ids))
    with NativeCapture(model, batch.image_grids[0]) as capture:
        processor = ReadoutProcessor(capture, bridge, parser, len(prompt))
        processor.capture_logits = capture_logits
        with torch.no_grad():
            result = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                repetition_penalty=1.0, eos_token_id=tokenizer.convert_tokens_to_ids('<|im_end|>'),
                pad_token_id=tokenizer.pad_token_id, use_cache=True,
                logits_processor=LogitsProcessorList([processor]))
    tokens = result[0, len(history):].tolist()
    return dict(token_ids=tokens, trace=processor.trace, saved_logits=processor.saved_logits,
                text=tokenizer.decode(tokens, skip_special_tokens=False), cap=max_new_tokens,
                stop_reason='im_end' if tokens and tokens[-1] == tokenizer.convert_tokens_to_ids('<|im_end|>') else 'length')
