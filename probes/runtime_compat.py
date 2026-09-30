"""Scope-bound HF native-generation/backward compatibility diagnostic.

Extracted from the 2026-09-29 saved probe. This is not a vLLM benchmark or an
experiment continuation. --check-inputs is CPU-only; execution requires clean
current source, explicit input files, the existing policy and separate GPU authority.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'inputs', 'retained', 'encodings', 'policy', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--image-id', type=int, default=7116)
    parser.add_argument('--check-inputs', action='store_true')
    return parser.parse_args(argv)


def inspect_inputs(args):
    if args.output.exists():
        raise FileExistsError('output exists; refusing overwrite')
    if not args.output.parent.is_dir() or not args.checkpoint.is_dir():
        raise ValueError('checkpoint and output-parent directories must exist')
    hashes = {}
    for name in ('inputs', 'retained', 'encodings', 'policy'):
        path = getattr(args, name)
        data = path.read_bytes()
        value = json.loads(data)
        hashes[name] = dict(path=str(path.resolve()), sha256=hashlib.sha256(data).hexdigest())
        if name != 'policy':
            ids = [x['image_id'] for x in value]
            if len(ids) != len(set(ids)) or args.image_id not in ids:
                raise ValueError(f'{name}: duplicate or missing requested image')
    return hashes


def assert_clean_source(expected=None):
    root = Path(__file__).resolve().parents[1]
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()
    if git('status', '--porcelain', '--untracked-files=normal'):
        raise RuntimeError('model execution requires clean current source; help/input checks remain available')
    identity = dict(checkout=str(root), commit=git('rev-parse', 'HEAD'), tree=git('rev-parse', 'HEAD^{tree}'))
    if expected is not None and identity != expected:
        raise RuntimeError('source identity changed during probe')
    return identity


def run(args):
    input_context = inspect_inputs(args)
    source_context = assert_clean_source()
    import torch
    import transformers
    import peft
    import flash_attn
    from probes import iterative_positive as p, online_row_credit as o
    from src.losses.vocab import build_token_vocabulary_groups
    from src.qwen.generation import NativeGenerationPolicy, generate_continuations

    if args.policy.resolve() != p.POLICY.resolve():
        raise ValueError('this diagnostic uses the current iterative_positive policy; a new policy needs separate qualification')
    print('versions', torch.__version__, transformers.__version__, peft.__version__, flash_attn.__version__, flush=True)
    q, delta, receipt = p.compose(args.checkpoint, evaluation=False)
    print('composed', len(receipt['parameters']), torch.cuda.memory_allocated() / 2**30, flush=True)
    inputs = {x['image_id']: x for x in p.load(args.inputs)}
    images = {x['image_id']: x for x in p.load(args.retained)}
    encodings = {x['image_id']: x for x in p.load(args.encodings)}
    item = inputs[args.image_id]
    batch = o.native_batch(q, item)
    print('native', batch.prompt_token_ids[0].__len__(), batch.image_grids[0], flush=True)
    q.model.eval()
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        result = generate_continuations(q.model, batch, extensions=[()], budgets=[32],
            eos_token_id=q.tokenizer.convert_tokens_to_ids('<|im_end|>'),
            pad_token_id=q.tokenizer.pad_token_id,
            policy=NativeGenerationPolicy(temperature=0, top_p=1, top_k=0, repetition_penalty=1, use_model_defaults=False))[0]
    print('generated', len(result.token_ids), result.stop_reason, flush=True)
    q.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    q.model.enable_input_require_grads()
    q.model.train()
    vocab = build_token_vocabulary_groups(q.token_identity, tokenizer=q.tokenizer)
    plan = {'producer': 'compat-probe'}
    record = {'raw_identity': 'compat-probe', 'image_grid_thw': item['image_grid_thw']}
    loss, evidence = o.forward(q, q.model, batch, images[args.image_id], record, plan, encodings[args.image_id], vocab, 'R')
    assert torch.isfinite(loss).item()
    loss.backward()
    grads = [(name, float(param.grad.detach().float().norm())) for name, param in q.model.named_parameters()
             if param.requires_grad and param.grad is not None and torch.isfinite(param.grad).all() and param.grad.detach().abs().max() > 0]
    assert grads, 'no finite nonzero trainable gradients'
    report = {'versions': {'torch': torch.__version__, 'transformers': transformers.__version__, 'peft': peft.__version__, 'flash_attn': flash_attn.__version__},
              'checkpoint': str(args.checkpoint), 'image_id': args.image_id, 'prompt_tokens': len(batch.prompt_token_ids[0]),
              'generated_tokens': len(result.token_ids), 'generation_stop': result.stop_reason,
              'teacher_tokens': evidence['tokens'], 'loss': float(loss.detach()), 'grad_count': len(grads),
              'grad_examples': grads[:8], 'cuda_peak_gib': torch.cuda.max_memory_allocated() / 2**30}
    report['source_context'] = source_context
    report['explicit_inputs'] = input_context
    assert_clean_source(source_context)
    if input_context != inspect_inputs(args):
        raise RuntimeError('input changed during probe')
    with args.output.open('x') as stream:
        stream.write(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


def main(argv=None):
    args = parse_args(argv)
    if args.check_inputs:
        print(json.dumps(dict(status='inputs_readable_only', inputs=inspect_inputs(args)), indent=2))
    else:
        run(args)


if __name__ == '__main__':
    main()
