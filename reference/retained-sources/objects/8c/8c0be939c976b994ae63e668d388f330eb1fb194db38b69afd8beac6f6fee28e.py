"""Native full-prefix numerical-feedback replay; CPU owns episode/candidate selection."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from transformers import GenerationConfig, LogitsProcessor, LogitsProcessorList, StoppingCriteria, StoppingCriteriaList

from probes.training_set_completion.readout_norm_fresh import _binding, _input_identity
from probes.training_set_completion.untied_shared import load_model
from src.inference.bound_requests import build_bound_native_requests
from src.qwen.native import prepare_native_inputs, _STALE_HISTORY_FIELDS

EOS, COORD, END = 151645, 151670, 151649


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def full_prefix(batch, raw, offset, pad, device, mutation=None):
    """Preserve accepted full-batch EOS padding and attention-mask convention."""
    suffix = [list(r['token_ids'][:offset]) + [pad] * max(0, offset-len(r['token_ids'])) for r in raw]
    if mutation is not None:
        bi, position, old, new = mutation
        if not 0 <= position < offset or suffix[bi][position] != old:
            raise ValueError('mutation does not name the original prefix token')
        suffix[bi][position] = new
    inputs = {k:v for k,v in batch.inputs.items() if k not in _STALE_HISTORY_FIELDS
              and k not in ('use_cache', 'return_dict', 'logits_to_keep')}
    ext = torch.tensor(suffix, device=device, dtype=torch.long).reshape(len(raw), offset)
    inputs['input_ids'] = torch.cat((batch.inputs['input_ids'], ext), dim=1)
    inputs['attention_mask'] = torch.cat((batch.inputs['attention_mask'], torch.ones(
        len(raw), offset, device=device, dtype=batch.inputs['attention_mask'].dtype)), dim=1)
    return inputs


class Runtime:
    def __init__(self, args):
        self.args = args
        self.output = args.output
        self.output.mkdir(parents=True, exist_ok=False)
        self.started = time.monotonic()
        self.ledger = dict(status='running', pid=os.getpid(), model=args.model,
            mode=args.mode, model_forwards=0, vision_forwards=0, tensor_bytes=0,
            panel=_binding(args.panel), producer=_binding(Path(__file__)), artifacts=[])
        self.handles = []
        self.persist()
        self.panel = json.loads(args.panel.read_text())
        self.q, self.ledger['identity'] = load_model(args.model, args.device)
        self.model = self.q.model
        self.head = self.model.get_output_embeddings()
        ids = self.head.selected_token_ids
        if ids[4:].tolist() != list(range(COORD, COORD+1000)):
            raise ValueError('unexpected coordinate token IDs')
        self.ids = ids[4:]
        self.U = (self.head.base.weight[ids] + self.head.shared_embed_delta).detach()[4:]
        self.E = self.model.get_input_embeddings()(ids).detach()[4:]
        norms = self.U.double().norm(dim=1)
        self.factors = norms.median()/norms
        self.versions = {n:p._version for n,p in self.model.named_parameters()}
        self.last = {}
        def count(*_):
            self.ledger['model_forwards'] += 1
            if self.ledger['model_forwards'] > args.max_forwards or time.monotonic()-self.started > args.max_seconds:
                raise RuntimeError('allocated forward/time budget exhausted')
        def vision(*_): self.ledger['vision_forwards'] += 1
        def hidden(mod, inputs): self.last['head_input'] = inputs[0][:,-1,:].detach()
        def logits(mod, inputs, output): self.last['logits'] = output[:,-1,:].detach()
        self.handles = [self.model.register_forward_pre_hook(count),
            self.model.model.visual.register_forward_pre_hook(vision),
            self.head.register_forward_pre_hook(hidden), self.head.register_forward_hook(logits)]
        self.save('weights.pt', dict(coordinate_ids=self.ids.cpu(), input_rows=self.E.cpu(),
            output_rows=self.U.cpu(), factors=self.factors.cpu()))
        self.persist()

    def persist(self):
        self.ledger['gpu_seconds'] = time.monotonic()-self.started
        write(self.output/'receipt.json', self.ledger)

    def save(self, name, payload):
        path = self.output/name
        torch.save(payload, path)
        self.ledger['tensor_bytes'] += path.stat().st_size
        if self.ledger['tensor_bytes'] > self.args.max_bytes:
            raise RuntimeError('allocated tensor budget exhausted')
        self.ledger['artifacts'].append(_binding(path))
        return path

    def batch(self, group_key, source=None):
        group = next(g for g in self.panel['groups'] if g['key']==group_key)
        config = dict(self.panel['configs'][self.args.model])
        config['data'] = dict(input_jsonl=group['input_jsonl'])
        requests, _ = build_bound_native_requests(self.q, config, group['cases'])
        batch = prepare_native_inputs(self.q.processor, requests, device=self.args.device, record_media_identity=True)
        if source is not None:
            receipt = json.loads(Path(source['receipt']['path']).read_text())
            for k in ('raw','trace','receipt'):
                if source[k] != _binding(Path(source[k]['path'])):
                    raise ValueError('source binding changed')
            if receipt['status'] != 'candidate_complete' or receipt['input_identity'] != _input_identity(batch):
                raise ValueError('source input identity or completion mismatch')
            for field in ('input_rows_sha256','output_rows_sha256','input_delta_sha256','output_delta_sha256'):
                if receipt['identity'][field] != self.ledger['identity'][field]:
                    raise ValueError('source checkpoint/effective row mismatch')
        return group, batch

    def generate(self, inputs, cap, processors=None, stopping=None):
        config = GenerationConfig(max_new_tokens=cap, do_sample=False, repetition_penalty=1,
            eos_token_id=EOS, pad_token_id=self.q.tokenizer.pad_token_id)
        with torch.inference_mode():
            return self.model.generate(**inputs, generation_config=config, use_model_defaults=False,
                logits_processor=LogitsProcessorList(processors or []),
                stopping_criteria=StoppingCriteriaList(stopping or []))

    def replay(self, batch, raw, trace, offset, bi, mutation=None):
        inputs = full_prefix(batch, raw, offset, self.q.tokenizer.pad_token_id, self.args.device, mutation)
        self.last.clear()
        self.generate(inputs, 1)
        packet = {k:v[bi].cpu().clone() for k,v in self.last.items()}
        if mutation is None:
            top = packet['logits'].topk(2)
            error = max(abs(float(top.values[k])-trace[offset]['raw_top2'][bi][k]) for k in (0,1))
            packet['parity'] = dict(top2_max_abs_error=error,
                winner_match=int(top.indices[0])==trace[offset]['raw_winners'][bi],
                tolerance=self.panel['tolerances']['logit_atol'])
            packet['parity']['passed'] = packet['parity']['winner_match'] and error <= self.panel['tolerances']['logit_atol']
        return packet

    def release(self, batch, raw, offset, bi, mutation=None, cap=512, row_cap=32):
        inputs = full_prefix(batch, raw, offset, self.q.tokenizer.pad_token_id, self.args.device, mutation)
        width = inputs['input_ids'].shape[1]
        class RowStop(StoppingCriteria):
            def __call__(self, ids, scores, **kwargs):
                target = ids[bi,width:]
                # Ending the complete batch does not alter any target decision.
                done = bool((target==EOS).any() or (target==END).sum() >= row_cap)
                return torch.full((ids.shape[0],), done, device=ids.device, dtype=torch.bool)
        result = self.generate(inputs, cap, stopping=[RowStop()])
        tokens = result[bi,width:].tolist()
        if EOS in tokens: tokens = tokens[:tokens.index(EOS)+1]
        return dict(token_ids=tokens, text=self.q.tokenizer.decode(tokens, skip_special_tokens=False,
            clean_up_tokenization_spaces=False), stop='eos' if EOS in tokens else
            'rows' if tokens.count(END)>=row_cap else 'cap', complete_rows=tokens.count(END),
            prefix_offset=offset, mutation=mutation, row_cap=row_cap, token_cap=cap)

    def natural(self):
        group, batch = self.batch(self.args.group)
        self.ledger['input_identity'] = _input_identity(batch)
        self.ledger.update(condition=f'{self.args.model}-{self.args.policy}', group=self.args.group)
        width = batch.inputs['input_ids'].shape[1]
        traces = []
        runtime = self
        class Policy(LogitsProcessor):
            def __call__(self, tokens, scores):
                transformed = scores.clone()
                transformed[:,runtime.ids] = (scores[:,runtime.ids].double()*runtime.factors).to(scores.dtype)
                used = transformed if runtime.args.policy=='normalized' else scores
                top = scores.topk(2)
                chosen = used.argmax(-1)
                traces.append(dict(offset=tokens.shape[1]-width, raw_winners=top.indices[:,0].tolist(),
                    raw_top2=top.values.tolist(), raw_runnerups=top.indices[:,1].tolist(), chosen=chosen.tolist(),
                    eos_logits=scores[:,EOS].tolist(), logsumexp=scores.logsumexp(-1).tolist(),
                    chosen_raw_logits=scores.gather(1,chosen[:,None]).squeeze(1).tolist()))
                return used
        result = self.generate(batch.inputs, 3084, processors=[Policy()])
        rows = []
        for case, tokens in zip(group['cases'], result[:,width:].tolist()):
            if EOS in tokens: tokens=tokens[:tokens.index(EOS)+1]
            rows.append(dict(image_id=int(case['input_record']['image_id']), row_id=case['row_id'],
                token_ids=tokens, text=self.q.tokenizer.decode(tokens,skip_special_tokens=False,
                    clean_up_tokenization_spaces=False), stop='eos' if EOS in tokens else 'length'))
        write(self.output/'raw.json', dict(rows=rows)); write(self.output/'trace.json', dict(steps=traces))
        if json.loads((self.output/'raw.json').read_text())['rows'] != rows: raise ValueError('raw readback failed')
        self.ledger.update(raw=_binding(self.output/'raw.json'), trace=_binding(self.output/'trace.json'))

    def close(self, error=None):
        for h in self.handles: h.remove()
        if hasattr(self, 'model'):
            if self.versions != {n:p._version for n,p in self.model.named_parameters()}:
                error = error or RuntimeError('parameter mutation detected')
        self.ledger.update(status='partial_candidate' if error else 'candidate_complete',
            error=repr(error) if error else None, live_jobs=[], ended_pid=os.getpid())
        self.persist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['natural','capture-source','paired'])
    parser.add_argument('--panel',type=Path,required=True)
    parser.add_argument('--model',choices=['tied','untied'],required=True)
    parser.add_argument('--group')
    parser.add_argument('--policy',choices=['original','normalized'],default='original')
    parser.add_argument('--selection',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--max-forwards',type=int,default=3300)
    parser.add_argument('--max-seconds',type=float,default=3600)
    parser.add_argument('--max-bytes',type=int,default=1024**3)
    args=parser.parse_args()
    runtime = Runtime.__new__(Runtime)
    try:
        runtime.__init__(args)
        if args.mode=='natural': runtime.natural()
        else: raise NotImplementedError('CPU selection interface not yet released')
    except BaseException as exc:
        if hasattr(runtime,'ledger'): runtime.close(exc)
        raise
    else: runtime.close()


if __name__=='__main__': main()
